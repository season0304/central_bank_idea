"""WCB 시간 방향을 학습하는 최소 예제.

0 = 비미래지향, 1 = 미래지향.
검증·조기 종료·재시작을 생략하고 Triplet 학습의 핵심만 보여준다.
"""

import math
import random
from contextlib import nullcontext
from itertools import cycle, islice
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 8                     # 한 배치에 비미래 4문장 + 미래 4문장
EPOCHS = 3
MARGIN = 0.2


def make_batches(rows):
    """두 라벨을 같은 수로 뽑는다. 적은 클래스는 순환해서 다시 사용한다."""
    groups = [[row for row in rows if row["label"] == label] for label in (0, 1)]
    for group in groups:
        random.shuffle(group)

    per_label = BATCH_SIZE // 2
    pools = [cycle(group) for group in groups]
    steps = math.ceil(max(len(group) for group in groups) / per_label)

    for _ in range(steps):
        yield list(islice(pools[0], per_label)) + list(islice(pools[1], per_label))


def triplet_loss(embeddings, labels):
    """각 문장에 대해 가장 어려운 positive와 negative를 배치 안에서 선택한다."""
    # 1. 벡터 길이를 1로 맞추면 내적이 코사인 유사도가 된다.
    vectors = F.normalize(embeddings.float(), p=2, dim=1)
    distances = (1 - vectors @ vectors.T).clamp(0, 2)

    # 2. 같은 라벨은 positive, 다른 라벨은 negative. 자기 자신은 제외한다.
    same_label = labels[:, None] == labels[None, :]
    self_pair = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_label & ~self_pair
    negative_mask = ~same_label

    # 3. 같은 라벨 중 가장 먼 문장, 다른 라벨 중 가장 가까운 문장을 고른다.
    hardest_positive = distances.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, torch.inf).min(dim=1).values

    # 4. 다른 라벨과의 거리가 같은 라벨보다 margin만큼 커지도록 학습한다.
    return F.relu(hardest_positive - hardest_negative + MARGIN).mean()


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    # 1. 이미 전처리된 WCB 학습 데이터: sentence와 label을 읽는다.
    rows = pq.read_table(
        ROOT / "data/processed/wcb_time_5768/train.parquet",
        columns=["sentence", "label"],
    ).to_pylist()

    # 2. 문장을 벡터로 바꾸는 모델을 불러온다.
    model = SentenceTransformer(
        str(ROOT / "models/gte-modernbert-base"),
        device=device,
        local_files_only=True,
        model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
        config_kwargs={"reference_compile": False},
    )
    model.max_seq_length = 256
    model[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.train()

    # 3. 문장 → 토큰 → 임베딩 → loss → 역전파 → 가중치 업데이트.
    for epoch in range(EPOCHS):
        losses = []
        for step, batch in enumerate(make_batches(rows), start=1):
            sentences = [row["sentence"] for row in batch]
            labels = torch.tensor([row["label"] for row in batch], device=device)
            features = model.preprocess(sentences)
            features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext():
                embeddings = model(features)["sentence_embedding"]
            loss = triplet_loss(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            if step % 100 == 0:
                print(f"epoch={epoch + 1}, step={step}, loss={loss.item():.4f}", flush=True)

        print(f"epoch={epoch + 1}, 평균 학습 loss={sum(losses) / len(losses):.4f}", flush=True)

    # 4. 학습된 문장 임베딩 모델을 저장한다.
    model.save_pretrained(str(ROOT / "runs/triplet-example"))


if __name__ == "__main__":
    main()
