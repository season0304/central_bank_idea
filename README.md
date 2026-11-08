# Central Bank Communication Classification

WCB 문장을 미래·비미래(`time_label`), 확실·불확실(`certain_label`), 정책 입장
(`stance_label`: 비둘기파·매파·중립·무관)으로 분류하는 연구 코드다.
GTE-ModernBERT-base의 임베딩 미세조정과 고정된 임베딩 위의 선형 분류기 학습을 포함한다.
현재 권장 실행 방법은 [세 작업 파이프라인 문서](wcb_4090/README.md)에 정리했다.

이 저장소에는 소스 코드·실행 안내·집계된 평가 결과를 포함한다.
참고 논문 PDF 7개는 [paper/](paper/)에 포함한다.
큰 WCB 논문은 이미지 해상도를 낮추지 않고 PDF 구조를 정리한 `.compressed.pdf`로
포함한다. 원본과 페이지 수(228쪽) 및 공백을 제외한 추출 텍스트가 같음을 확인했다.
원본 데이터, 가중치, 임베딩 캐시, 체크포인트는 Git에 포함하지 않는다.

## 빠른 시작

Python 3.12 환경에서:

```bash
uv sync --python 3.12
uv run python -m scripts.download_resources --seeds 5768
uv run python -m wcb_4090.prepare --task time_label --seed 5768
uv run python -m wcb_4090.prepare --task certain_label --seed 5768
uv run python -m wcb_4090.prepare --task stance_label --seed 5768
uv run python -m wcb_4090.pipeline \
  --tasks time_label certain_label stance_label \
  --output runs/wcb-stable-all \
  --device cuda --precision auto \
  --triplet-mining semi_hard --reference-weight 1 \
  --triplet-patience 0 --classifier-patience 0
```

다운로드 스크립트는 원본 GTE 리비전을 고정하고 실제 데이터 리비전도
`data/resource_downloads.json`에 기록한다. 데이터/모델 제공처의 접근 조건과
라이선스는 각 Hugging Face 저장소를 따른다. 준비·다운로드 단계에서는 모델을 학습하지 않는다.

## 기록된 테스트 결과

`5768` 분할에서 검증으로 선택한 모델의 테스트 결과다.

| 작업 | Accuracy | Macro-F1 |
|---|---:|---:|
| 미래·비미래 | 0.8658 | 0.8638 |
| 확실·불확실 | 0.8288 | 0.7546 |
| 정책 입장 | 0.6436 | 0.6369 |

확실·불확실과 정책 입장에서는 Triplet 미세조정 전 원본 GTE(step 0)가 최적
인코더로 선택됐다. 이 두 작업의 결과는 원본 GTE에 학습한 선형 분류기의 결과이며,
Triplet 미세조정의 성능 향상을 입증하는 수치로 해석하지 않는다.

- [세 모델의 집계 결과](reports/wcb-stable-all.json)
- [WCB 논문과 지표를 맞춘 비교](reports/paper-comparison.json)
- [같은 모델의 다른 시드 테스트 분할 평가](reports/cross-seed-evaluation.json)

다른 시드 테스트셋의 약 85%는 기존 `5768` 훈련·검증 문장과 겹친다.
교차 분할 보고서는 중복 포함/제외 점수를 모두 기록하며, 세 시드로 각각 학습한
모델들의 독립 실험 결과가 아니다. 보고서는 아래 명령으로 갱신할 수 있다.

```bash
python -m scripts.export_eval_reports
```

데이터와 원논문: [WCB 데이터셋](https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000),
[Words That Unite The World](https://arxiv.org/abs/2505.17048),
[원본 GTE 모델](https://huggingface.co/Alibaba-NLP/gte-modernbert-base).
IMF의 [From Text to Quantified Insights](https://www.imf.org/-/media/files/publications/wp/2025/english/wpiea2025109-print-pdf.pdf)는
참고 연구이며, 현재 구현을 해당 논문의 완전한 재현으로 간주하지 않는다.

## 이전 단일 시간 라벨 실험 기록

첫 작업은 `time_label`만 사용한다. 다른 라벨과 결합하지 않는다.

- `not forward looking` → `0` (과거 및 현재)
- `forward looking` → `1` (미래)

원본 WCB의 `5768` 분할을 사용한다. 세 시드 폴더를 합치지 않는다.
WCB 데이터는 CC-BY-NC-SA-4.0이며 원 저자와 논문을 인용해야 한다.
출처: https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000

## 1. 데이터 준비

```bash
uv sync --python 3.12
uv run python -m wcb_triplet.prepare
```

원문은 수정하지 않는다. 공백 차이만 제거한 비교 키로 중복을 확인한다.
동일 텍스트에 서로 다른 시간 라벨이 붙은 그룹은 모두 제외한다.
나머지 동일 텍스트는 `test > val > train` 우선순위로 한 건만 유지해
학습과 평가 사이의 동일 문장 중복을 없앤다. 제외 기록과 원본 체크섬은
`data/processed/wcb_time_5768/`에 저장한다.
원본 라벨 파일에는 문서 ID가 없어 문서 단위 분리와 앞뒤 문장 복원은
아직 수행하지 않았다. 의미상 유사한 문장 사이의 누출도 별도 검토가 필요하다.

## 2. 배치와 손실 연결 확인

```bash
uv run python -m wcb_triplet.inspect_batch
```

실제 WCB 문장 8개(클래스별 4개)를 추출하고, **인공 벡터**로
`BatchHardTripletLoss` 계산과 역전파가 동작하는지 확인한다.
출력 JSON의 loss는 분류 성능이나 BGE-M3의 학습 손실이 아니다.
각 기준 문장에 선택된 같은 라벨/다른 라벨 문장을 JSON에서 확인할 수 있다.

실제 학습에서는 `make_loss(encoder)`에 GTE-ModernBERT-base를 넘기고
두 라벨이 같은 수로 들어가는 `BalancedBatchSampler`를 사용한다.
`margin=0.2`는 예시 초기값이며 IMF 논문의 실제 값으로 확인된 것이 아니다.
학습 코드는 아래 4절과 같다. 현재 구현은 임베딩 파인튜닝 단계이며,
후속 dense/softmax 분류기 학습은 포함하지 않는다.

## 3. 내려받은 모델 불러오기

`Alibaba-NLP/gte-modernbert-base`를 `models/gte-modernbert-base/`에 저장했다.
다운로드한 리비전은 `e7f32e3c00f91d699e8c43b53106206bcc72bb22`이며,
추론과 파인튜닝에 필요한 파일 9개의 합계는 301,662,255바이트(약 302MB)다.
가중치 파일은 Hub에서 제공하는 SHA-256과 일치하는지 확인했다.
리비전과 파일별 체크섬은 모델 폴더의 `download_manifest.json`에 기록했다.

```python
import torch
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(
    "models/gte-modernbert-base",
    device="cuda",
    local_files_only=True,
    model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
    config_kwargs={"reference_compile": False},
)
encoder.max_seq_length = 256
```

오프라인으로 모델을 불러온 뒤 RTX 3060 Laptop GPU에서 문장 3개의
768차원 임베딩 생성을 확인했다. 상세 결과는 `load_check.json`에 기록했다.
이는 모델 다운로드 당시의 로딩 및 추론 확인 기록이다.

## 4. WCB 시간 라벨로 임베딩 파인튜닝

새 학습에는 [안정화된 세 작업 파이프라인](wcb_4090/README.md)을 사용한다.
`wcb_4090.train`과 `wcb_4090.pipeline`은 semi-hard mining과 원본 임베딩 보존을
기본으로 사용한다. 아래 `wcb_triplet` 절은 기존 hard-only 실험과 재시작을 위한
기록이며, 해당 설정에서 관찰된 임베딩 압축을 해결한 새 설정은 위 문서에 설명한다.

프로젝트 루트에서 실행한다. 데이터와 모델은 위 단계에서 준비한 로컬 파일을
사용하므로 학습 중 Hugging Face 접속은 필요하지 않다.

```bash
uv run python -m wcb_triplet.train \
  --output runs/wcb-time \
  --device cuda \
  --precision bf16
```

기본 설정은 3 epochs, batch size 32, 입력 길이 256, AdamW 학습률 2e-5,
weight decay 0.01, warmup 10%, cosine-distance triplet margin 0.2다.
모델 전체를 파인튜닝한다. BF16 autocast와 gradient checkpointing을 지원하고,
거리 및 triplet loss 계산은 float32로 수행한다. 이는 우리 실험의 시작 설정이며
IMF 논문의 세부 설정을 완전히 재현한 것으로 간주하지 않는다.

각 배치에는 라벨별 16문장이 들어가고 같은 원본 행이 한 배치에서 반복되지 않는다.
매 epoch에 다수 클래스 전체를 사용하며 소수 클래스와 마지막 배치는 셔플 순환으로
채운다. 따라서 **소수 클래스 재표집(oversampling)**이 적용된다. 마지막 배치의
크기도 고정된다. epoch별 셔플은 seed와 epoch 번호로 재현한다.

검증에는 `val.parquet`만 사용한다. 검증도 두 클래스를 균등하게 구성하고 부족한
부분을 순환으로 채우며, **매번 같은 배치와 후보 문장**으로 전체 검증 데이터를
평가한다. 100스텝마다, epoch 끝, 마지막 스텝에 검증한다. 평균 검증 triplet loss가
가장 낮은 모델을 선택한다. 학습 전 모델도 최적 모델 후보에 포함한다.
이 검증값은 분류 accuracy/F1이 아니다. `test.parquet`은 학습 코드에서 읽지 않는다.

출력 파일은 다음과 같다.

```text
runs/wcb-time/
  config.json                  고정된 학습 설정
  data_manifest.json           데이터/원본 모델 체크섬 및 표집 방식
  events.jsonl                 매 스텝 손실·학습률·gradient norm·배치별 라벨 수
  best_model/                  검증 손실이 가장 낮은 SentenceTransformer
    wcb_task.json              라벨 정의, 선택된 스텝과 검증 손실
  checkpoints/step-*.pt        모델·AdamW·scheduler·난수·다음 배치 위치
  latest_checkpoint.json       마지막으로 완전히 저장된 체크포인트 경로
  summary.json                 실행 종료 상태 및 최적 모델 정보
```

체크포인트는 100스텝마다, 검증 최고 기록 갱신 시, 정상 중단/종료 시 저장한다.
임시 파일을 완전히 쓴 뒤 원자적으로 교체하고 최신 2개를 유지한다.
`events.jsonl`은 매 스텝 기록하며 터미널 출력은 기본 10스텝 간격이다.
새 프로세스마다 session ID를 붙이고 재시작 시 기존 로그에 이어 쓴다.

## 5. 중단 및 재시작

본 학습 설정을 유지하면서 10스텝만 먼저 실행해 볼 수 있다.
`--stop-after-steps`는 이번 실행에서 추가로 수행할 스텝 수이며 전체 학습 계획과
학습률 스케줄을 바꾸지 않는다.

```bash
uv run python -m wcb_triplet.train \
  --output runs/wcb-time \
  --device cuda \
  --precision bf16 \
  --stop-after-steps 10
```

이어서 실행할 때는 저장된 설정을 자동으로 불러온다.

```bash
uv run python -m wcb_triplet.train \
  --output runs/wcb-time \
  --resume \
  --device cuda
```

모델, optimizer, scheduler, Python/PyTorch/CUDA 난수 상태, epoch 및 다음 배치 위치를
복원한다. 정확한 재시작을 위해 배치 크기·학습률·epoch 수 등 학습 설정 변경은
거부한다. 데이터와 초기 모델 파일이 변경된 경우에도 체크섬 검사로 재시작을
거부한다. 원래 모델 폴더와 데이터도 함께 유지해야 한다.

Ctrl+C 또는 SIGTERM은 현재 스텝을 끝낸 뒤 저장하고 종료한다. 강제 종료나 호스트
장애 시에는 마지막 완성 체크포인트부터 복원하므로 그 이후 스텝은 다시 계산한다.
이 경우 로그의 session ID와 resume 이벤트로 재실행 구간을 구분할 수 있다.
동일 출력 폴더에 두 프로세스가 동시에 쓰는 것도 차단한다.

저장된 최적 모델은 다음처럼 불러온다.

```python
encoder = SentenceTransformer("runs/wcb-time/best_model", local_files_only=True)
embeddings = encoder.encode(sentences, normalize_embeddings=True)
```

이 출력은 문장 임베딩이다. 미래/비미래 예측 확률을 얻으려면 후속 분류기를 학습해야 한다.

## 6. Vast.ai 전송 및 실행

코드, 전처리 데이터, 초기 모델만 압축한다. 기존 원문 PDF/대용량 원천 데이터,
로컬 가상환경, 실험 결과, Hugging Face 캐시는 제외한다.

```bash
python scripts/package_vast.py
```

생성물은 `dist/wcb-vast.tar.gz`다. Vast.ai의 Connect에 표시되는 HOST와 PORT를
사용해 로컬 터미널에서 전송한다.

```bash
scp -P PORT dist/wcb-vast.tar.gz root@HOST:/workspace/
```

서버에서 다음을 실행한다.

```bash
mkdir -p /workspace/wcb
tar -xzf /workspace/wcb-vast.tar.gz -C /workspace/wcb
cd /workspace/wcb
```

Python 3.12와 CUDA 12.8 호환 드라이버를 갖춘 PyTorch 인스턴스에서 검증 환경을
맞추려면 다음을 사용한다. 서버에 호환 버전이 이미 설치되어 있으면 유지해도 된다.

```bash
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'sentence-transformers==6.2.0' 'transformers==5.22.2' 'pyarrow==26.0.1'
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

SSH 연결이 끊겨도 실행이 유지되도록 `tmux` 세션에서 시작한다.

```bash
tmux new -s wcb
python -m wcb_triplet.train --output runs/wcb-time --device cuda --precision bf16
```

세션에서 빠져나오려면 Ctrl+B 다음 D, 돌아오려면 `tmux attach -t wcb`를 사용한다.
서버에서 재시작 명령은 다음과 같다.

```bash
python -m wcb_triplet.train --output runs/wcb-time --resume --device cuda
```

다른 서버로 옮겨 재시작하려면 프로젝트와 함께 `runs/wcb-time/` 전체를 복사한다.
최적 모델만으로는 optimizer/scheduler 상태를 복원할 수 없다.

## 7. 동작 검증

```bash
uv run python -m unittest discover -s tests -v
```

인터넷 접속 없이 작은 로컬 인코더를 생성해 다음을 검증한다.

- 클래스별 균형, 배치 내 행 중복 방지, 전체 행 포함, seed 재현성
- 연속 학습과 중단·재시작 학습의 최종 가중치·optimizer·scheduler·난수 상태 일치
- 실제 가중치 갱신, 최적 모델 재로딩, 로그 및 체크포인트 보존 개수
- 설정/데이터 변경 시 재시작 거부 및 동시 쓰기 차단
- 지원 GPU가 있으면 CUDA BF16 학습 및 체크포인트 재시작

실제 GTE-ModernBERT와 WCB 데이터로도 CPU에서 1스텝 학습 후 저장하고, 재시작해
총 2스텝까지 완료했다. 기록은 `runs/local-gte-smoke/`에 있다. 이는 기능 검증이며
본 학습이나 분류 성능 평가가 아니다. 다른 하드웨어/라이브러리 버전 사이의
부동소수점 결과까지 완전히 같음을 보장하는 테스트는 아니다.

검증한 환경: Python 3.12, PyTorch 2.11.0+cu128,
Sentence Transformers 6.2.0, Transformers 5.22.2, PyArrow 26.0.1.
GPU를 사용할 때는 CUDA 빌드와 드라이버가 호환되는지 확인한다.
