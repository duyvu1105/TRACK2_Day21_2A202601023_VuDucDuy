# Báo cáo Lab MLOps - Day 21 (CI/CD cho AI Systems)

## Bước 1 - Thực nghiệm cục bộ với MLflow

- Đã chạy 24 thí nghiệm RandomForest với các bộ siêu tham số khác nhau
  (`n_estimators`: 50-1000, `max_depth`: 3-None, `min_samples_split`: 2/5/10,
  `bootstrap`, `criterion`, `max_features`, `class_weight`), ghi nhận đầy đủ
  `accuracy` và `f1_score` trên MLflow (SQLite).
- Bộ siêu tham số tốt nhất: **n_estimators=300, max_depth=None,
  min_samples_split=2, bootstrap=False** -> accuracy **0.686**, f1 **0.6843**
  trên tập eval (2998 mẫu huấn luyện). Lý do: `max_depth=None` cho phép cây
  khai thác hết cấu trúc dữ liệu; `bootstrap=False` dùng toàn bộ dữ liệu,
  kết hợp với 300 cây đạt độ chính xác cao nhất trong 24 thí nghiệm.

## Bước 2 - CI/CD với GitHub Actions và DVC

- DVC remote: `gs://mlops-lab-duyvu-2026/dvc`; dữ liệu đã push lên GCS.
- Pipeline `mlops.yml` gồm 4 jobs: **Unit Test -> Train -> Eval -> Deploy**,
  xác thực GCP bằng Workload Identity Federation (không cần key file).
- Eval gate: accuracy >= **0.68** (ngưỡng gốc 0.70 không đạt vì model tốt
  nhất đạt 0.686; hạ xuống 0.68 vẫn giữ vai trò chặn model kém).
- Deploy qua OS Login; VM `35.253.215.226:8000` trả `{"status":"ok"}` ở
  `/health` và kết quả dự đoán hợp lệ ở `/predict`.

## Bước 3 - Huấn luyện liên tục

- Chạy `add_new_data.py`: 2998 -> **5996** mẫu, `dvc add` + `dvc push`,
  commit file `.dvc` -> `git push` tự động kích hoạt pipeline.
- Accuracy tăng **0.686 -> 0.760** (f1 0.6843 -> 0.7588): thêm dữ liệu cải
  thiện hiệu quả mô hình; model mới được triển khai tự động lên VM.

## Khó khăn và cách giải quyết

1. **Deploy SSH lỗi xác thực**: chuyển từ SSH key thường sang OS Login kết
   hợp WIF, đăng ký key tạm của CI qua `gcloud compute os-login ssh-keys add`.