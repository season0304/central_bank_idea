# RTX 4090에서 WCB 세 모델을 한 번에 학습

기존 `wcb_triplet/`과 별도로 사용하는 파이프라인이다. Vast.ai의 한 GPU에서 세 작업의
Triplet 단계를 순서대로 실행하고, 결과를 내려받은 뒤 로컬에서 분류기를 학습한다.
각 작업은 동일한 원본 GTE-ModernBERT-base에서 새로 시작한다.
기존 작업의 가중치를 다음 작업의 초기값으로 넘기지 않는다.

| 작업 | 숫자 라벨 |
|---|---|
| `time_label` | 비미래 0, 미래 1 |
| `certain_label` | 확실 0, 불확실 1 |
| `stance_label` | 비둘기파 0, 매파 1, 중립 2, 무관 3 |

이 숫자 매핑은 이 프로젝트의 정의이며, 다른 Hugging Face 모델의 LABEL_0 등과
동일하다고 가정하면 안 된다. 원본 WCB의 문자열 라벨에서 변환한다.

각 작업의 학습 순서는 다음과 같다.

1. **Triplet 단계:** GTE 전체를 Batch Semi-Hard Triplet Loss로 미세조정하고,
   원본 GTE의 문장별 벡터 방향을 보존하는 항을 함께 학습한다.
   코사인 거리, margin 0.2, 기본 배치 32, 문장 길이 256, 최대 3 epochs다.
   학습률 2e-5, AdamW, BF16 자동 선택, gradient checkpointing을 사용한다.
   두 라벨은 배치당 16개씩, 네 라벨은 8개씩 넣는다. 부족한 클래스는 재표집한다.
2. **분류기 단계:** 가장 좋은 Triplet encoder를 고정하고 정규화된 임베딩을 만든다.
   단일 Dense 층을 Cross-Entropy로 학습한다. Softmax는 추론 때 적용한다.
   원래 학습 클래스 분포를 사용하며 encoder는 이 단계에서 업데이트하지 않는다.
3. **평가:** 검증으로 모델 선택을 마친 뒤 test를 한 번 평가한다.
   accuracy, macro-F1, 클래스별 precision/recall/F1과 confusion matrix를 저장한다.

분류기는 기본적으로 정규화된 임베딩의 각 차원에서 **훈련 데이터의 평균을 빼고
훈련 데이터의 표준편차로 나눈다**(`--feature-scaling standard`). 표준편차의 최솟값은
1e-6이다. 이 통계는 검증/test에서 다시 계산하지 않고 `best_classifier.pt`와
체크포인트에 저장하여 평가와 추론에도 적용한다. Triplet 학습 후 벡터 간 차이가
작아지면 이 과정 없이 분류기가 다수 클래스만 예측할 수 있다. 비교 실험에는
`--feature-scaling none`을 사용한다. 이전 실행의 config에 이 항목이 없으면 재시작 시
기존 방식인 `none`을 유지한다. 기존 실행에 표준화를 추가하려면 새 출력 폴더를 쓴다.

새 Triplet 실행의 기본 손실은 다음과 같다.

```text
loss = semi_hard_triplet_loss + reference_weight × mean(1 − cosine(student, original_gte))
```

`--triplet-mining semi_hard --reference-weight 1`이 기본값이다. 같은 클래스의 모든
문장 쌍을 사용하고, 각 양성 거리보다 먼 음성 후보 중 가장 가까운 것을 고른다.
그런 음성 후보가 없으면 라이브러리 구현은 가장 먼 음성 후보로 대체한다.
가장 먼 양성과 가장 가까운 음성만 강제하는 기존 방식의 압축을 줄이면서,
보존 항으로 원본 모델의 문장별 정보를 유지한다.
이는 [FaceNet의 semi-hard mining](https://www.florian-schroff.de/publications/schroff_facenet_v3.pdf)과
[Sentence Transformers 구현](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#batchsemihardtripletloss)을
사용한 이 프로젝트의 안정화 설정이다. 원 논문의 재현 설정으로 간주하지 않는다.

원본 기준 벡터는 미세조정 **전에** train/val에서만 계산하고
`reference_embeddings_train.pt`, `reference_embeddings_val.pt`에 캐시한다.
입력 파일 해시·문장 ID 순서·길이·정밀도가 맞아야 캐시를 재사용한다.
중단 후 캐시가 없으면 체크포인트 가중치를 불러오기 전에 원본 모델에서 다시 생성한다.
기준 벡터에는 역전파하지 않으며, 별도 원본 모델을 GPU 메모리에 계속 올려두지 않는다.

검증 `val_loss`와 모델 선택에는 위 두 손실의 합을 사용하고,
`triplet_loss`와 `reference_loss`를 따로 기록한다. 기존 hard-only 실행과 새 실행의
`val_loss`를 직접 비교하지 않는다. 벡터 기하와 분류 검증 성능도 함께 확인한다.
기존 설정에 새 항목이 없으면 `--resume`은 `hard`, 보존 가중치 `0`을 유지한다.
기존 체크포인트를 새 손실로 이어 붙이지 않고 새 출력 폴더와 원본 GTE에서 시작한다.

Triplet은 검증 loss가 0.001 이상 좋아지지 않는 검증이 5회 연속이면 멈춘다.
검증은 200 steps마다와 epoch 끝에 하며, 시작 전 모델도 최적 모델 후보에 포함한다.
검증 후보/배치 크기를 고정해 loss를 비교한다. 분류기는 기본적으로 검증 macro-F1이
0.0001 넘게 개선되지 않는 epoch가 5회면 멈춘다(최대 50 epochs).
가장 좋은 모델 자체는 작은 개선도 저장한다. `--selection-metric accuracy`로
분류기의 선택 기준을 바꿀 수 있다. 원 논문의 상세 설정을 그대로 재현한 실험은 아니다.

조기 종료를 모두 끄려면 pipeline에 `--triplet-patience 0 --classifier-patience 0`을
지정한다. 분류기 단독 실행은 `--early-stopping-patience 0`, 로컬 분류기 실행은
`--patience 0`을 사용한다. 검증과 최적 모델 저장은 계속 수행한다.

Triplet 검증 로그에는 `mean_pairwise_cosine`, `hard_positive_distance`,
`hard_negative_distance`, `triplet_satisfied_fraction`도 기록한다. 평균 유사도가 1에
가까워지고 두 거리가 모두 0에 가까워지면 loss가 margin 0.2에 접근하더라도
문장 간 구별이 좋아졌다는 뜻은 아니다. 이 통계는 고정된 균형 배치 기준이며
전체 검증 집합의 모든 문장 쌍 통계와는 다르다.
`triplet_satisfied_fraction`은 mining 설정과 관계없이 가장 먼 양성/가장 가까운
음성의 margin 조건을 진단한다. semi-hard 손실에서 실제 선택된 쌍의 만족 비율은 아니다.

로컬에서 세 작업의 임베딩·분류기·테스트 평가를 순서대로 실행하려면:

```bash
python -m wcb_4090.pipeline \
  --tasks time_label certain_label stance_label \
  --output runs/wcb-stable-all \
  --device cuda --precision auto \
  --triplet-mining semi_hard --reference-weight 1 \
  --triplet-patience 0 --classifier-patience 0
```

각 작업은 원본 GTE에서 시작한다. 위 명령은 조기 종료 없이 임베딩 3 epochs,
분류기 50 epochs를 수행한다. `--triplet-only`를 추가하면 임베딩 단계만 실행한다.

이미 저장된 train/val 임베딩을 점검하려면 다음 명령을 사용한다. 원본 GTE를 사용한
분류기 캐시가 있으면 `--baseline <분류기 폴더>`로 같은 문장끼리 비교할 수 있다.
이 점검은 test 데이터를 읽지 않는다.

```bash
python -m wcb_4090.audit \
  --classifiers runs/wcb-time/classifier \
    runs/wcb-other-two-full/certain_label/classifier \
    runs/wcb-other-two-full/stance_label/classifier \
  --output runs/training-audit/audit.json
```

## 로컬 데이터 준비 및 압축

프로젝트 루트에서 기존 Python 환경 또는 아래 패키지가 설치된 환경을 사용한다.

```bash
python -m wcb_4090.prepare --task time_label
python -m wcb_4090.prepare --task certain_label
python -m wcb_4090.prepare --task stance_label
python -m wcb_4090.package
```

전처리 결과는 별도 `data/processed_4090/`, 전송 파일은 `dist/wcb-4090.tar.gz`다.
WCB `5768` 분할만 사용한다. 라벨별로 중복과 라벨 충돌을 점검하여 제외 기록과
원본 SHA-256을 남긴다. 같은 문장은 `test > val > train` 우선순위로 하나만 보존한다.
원문은 수정하지 않으며 공백 차이를 무시한 동일 문장만 확인한다. 문서 ID가 없어
문서 단위 누출과 의미가 비슷한 문장 사이의 누출까지 배제하지는 못한다.
WCB의 CC-BY-NC-SA-4.0 조건과 원 저자 인용을 따라야 한다.

## Vast.ai에서 실행

Python 3.12와 CUDA 지원 PyTorch 환경을 사용한다. 모델·체크포인트·패키지 설치 공간을
위해 40GB 이상 디스크를 권장한다. 이 패키지는 PDF나 전체 연설문 데이터를 포함하지 않는다.
로컬에서 SSH 접속 정보의 실제 PORT/HOST로 바꾸어 전송한다.

```bash
scp -P PORT dist/wcb-4090.tar.gz root@HOST:/workspace/
```

Vast 서버에서:

```bash
mkdir -p /workspace/wcb
tar -xzf /workspace/wcb-4090.tar.gz -C /workspace/wcb
cd /workspace/wcb
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'sentence-transformers==6.2.0' 'transformers==5.22.2' 'pyarrow==26.0.1'
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
tmux new -s wcb4090
bash wcb_4090/train_vast.sh
```

GPU 드라이버는 Vast 호스트가 관리한다. 컨테이너 안에서 드라이버를 설치하지 않는다.
CUDA 12.8 PyTorch가 동작하는 호스트가 필요하다. RTX 4090의 실제 메모리 사용량은
다른 프로세스와 입력 길이에 따라 달라진다. 메모리가 부족하면 새 출력 폴더에서
`--batch-size 16 --embedding-batch-size 32` 등으로 시작한다. 배치를 바꾸면
batch-hard의 비교 후보도 달라지므로 동일 실험의 재시작 중에는 변경할 수 없다.

`Ctrl+B` 다음 `D`로 tmux에서 나와도 학습은 유지된다.
`tmux attach -t wcb4090`으로 돌아간다.

이 명령은 세 작업의 **Triplet 학습만** 수행한다. 분류기까지 서버에서 모두 실행하려면
`python -m wcb_4090.pipeline`을 직접 사용한다. 두 방식은 서로 다른 출력 폴더에서
시작해야 하며 재시작 중에 단계를 변경하지 않는다.

## 재시작

```bash
bash wcb_4090/train_vast.sh --resume
```

기본값을 바꿔 시작했다면 동일한 옵션들을 `--resume`과 함께 지정한다.
실행 중인 단계는 체크포인트부터 이어지고, 끝난 단계는 저장 상태를 검증한 뒤 다음으로 간다.
모델·데이터 SHA-256이 달라지면 재시작을 거부한다. 같은 결과 폴더의 동시 실행도 차단한다.
Triplet은 optimizer/scheduler/RNG/배치 위치/조기 종료 카운터를 복원한다.
분류기는 optimizer/RNG/epoch/배치 위치/조기 종료 카운터를 복원한다.
SIGINT/SIGTERM은 현재 배치를 마친 뒤 저장한다. 임베딩 캐시 생성 도중 중단되면
완료된 split 캐시만 재사용한다. 호스트 강제 종료 시에는 마지막 저장 이후 배치를 다시 수행한다.

## 결과와 추론

Vast에서 세 Triplet 학습이 끝나면 로컬 터미널에서 결과를 내려받는다.
`PORT`와 `HOST`는 실제 Vast SSH 정보로 바꾼다. 현재 프로젝트 루트에서 실행한다.

```bash
mkdir -p runs
scp -P PORT -r root@HOST:/workspace/wcb/runs/wcb-4090-all runs/
```

체크포인트 전송량을 줄이려면 각 작업의 `triplet/best_model/`과
루트 `pipeline_config.json`만 받아도 로컬 분류기를 학습할 수 있다.
작업별 경로 구조 `runs/wcb-4090-all/<task>/triplet/best_model/`을 유지한다.
각 best_model에는 가중치, tokenizer, pooling 설정 및 `wcb_task.json`이 모두 필요하다.

로컬 환경에 torch/sentence-transformers/transformers/pyarrow가 설치되어 있어야 한다.
현재 노트북에서는 아래처럼 기존 가상환경의 Python을 지정할 수 있다.

```bash
/home/sungju/workspace/.venv-qwen-embedding/bin/python -m wcb_4090.local_classifiers \
  --run runs/wcb-4090-all --device cpu
```

기본값은 CPU다. encoder는 고정하고 train/val 임베딩을 먼저 생성·캐시하므로
별도 GPU가 없어도 실행된다. CPU에서 긴 시간이 드는 부분은 이 임베딩 생성이며,
그 뒤 작은 Dense 분류기만 학습한다. 캐시는 재시작 때 다시 사용한다.
여유 있는 로컬 GPU를 사용하려면 `--device cuda --precision bf16`을 지정한다.
로컬에서 중단 후 이어가기:

```bash
python -m wcb_4090.local_classifiers --run runs/wcb-4090-all --device cpu --resume
```

이 명령은 Triplet 학습을 다시 실행하지 않는다. 분류기 완료 후에만 테스트 데이터를
평가한다. 로컬 결과는 같은 run 폴더의 각 작업 아래 `classifier/`에 저장되며,
요약은 `local_classifiers_summary.json`에 기록한다.

```text
runs/wcb-4090-all/
  pipeline_summary.json          서버 Triplet 실행 요약
  local_classifiers_summary.json 로컬 분류기 및 평가 요약
  time_label/
    triplet/best_model/          최적 SentenceTransformer
    triplet/checkpoints/         optimizer 포함 체크포인트
    classifier/best_classifier.pt
    classifier/test_metrics.json
  certain_label/                 동일 구조
  stance_label/                  동일 구조
```

임베딩 캐시는 각 classifier 폴더에 저장된다. 모델 선택에는 train/val만 사용하며
test 결과를 튜닝에 반복 사용하지 않는다. test 결과가 있으면 입력 해시를 확인하고 재사용한다.
서버를 삭제하기 전에 전체 결과 폴더를 내려받는다. 추론에 필요한 것은 각 작업의
`triplet/best_model/`, `classifier/best_classifier.pt`, `classifier/config.json`과
루트 `pipeline_config.json`이다. 경로를 옮겨도 추론기는 같은 작업 폴더의 encoder를 찾는다.

```bash
python -m wcb_4090.predict --run runs/wcb-4090-all \
  --text 'The Committee expects inflation to decline next year.'
```

점수는 Softmax 출력이며 정답 확률로 보정된 값은 아니다.

CPU 검증용 작은 모델 테스트:

```bash
python -m unittest discover -s wcb_4090/tests -v
```
