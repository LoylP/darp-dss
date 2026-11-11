# DARP-DSS: Tái hiện "Decision Aggregation with Reliability Propagation" và hệ thống hỗ trợ quyết định demo

## Giới thiệu

DARP tổng hợp các khuyến nghị của nhiều chuyên gia để hỗ trợ quyết định trong bối cảnh dự đoán có thể sai khác về độ tin cậy. Repo triển khai phương pháp trong bài báo “Decision aggregation with reliability propagation” (Zhong và cộng sự, 2024). Bên cạnh pipeline tái hiện, repo cung cấp một DSS demo để xem kết quả, truy vấn giải thích và thử nghiệm quyết định mới. Dữ liệu mặc định được mô phỏng để có thể chạy độc lập với các cơ sở dữ liệu tài chính có cấp phép.

## Điểm nổi bật

- Tái lập có kiểm soát bằng seed 2024; cùng cấu hình và môi trường sẽ sinh dữ liệu xác định.
- Kiểm chứng các tính chất lý thuyết bằng test: Laplacian có dấu nửa xác định dương, chuẩn hóa \(\eta\), và ví dụ minh họa trong bài báo.
- Hỗ trợ giải thích quyết định bằng điểm tin cậy \(c\) và các láng giềng ảnh hưởng.
- Các kết quả package hiện có sai khác nhỏ với một số giá trị trong fixture tham chiếu; chi tiết được trình bày ở phần kết quả.

## Kiến trúc

```text
Streamlit (ui/)
      │ HTTP/JSON
      ▼
FastAPI (api/)
      │
      ▼
package darp (src/darp/)
      │ đọc
      ▼
artifacts/ (sinh bằng make precompute)
```

| Thành phần DSS | Thành phần trong repo |
|---|---|
| Dữ liệu | `darp.data.synthetic`, `darp.data.loaders`, `darp.dataset` |
| Mô hình | `darp.relevance`, `darp.model`, `darp.baselines`, `darp.evaluation` |
| Tri thức | Đặc trưng f1–f6, tham số cấu hình và điểm tin cậy trong `darp` |
| Giao diện | `api/` cung cấp API; `ui/` trình bày kết quả và quyết định |

## Cấu trúc thư mục

```text
api/                    # API FastAPI và schema request/response
artifacts/              # Kết quả sinh tại máy khi chạy precompute
configs/                # Cấu hình mặc định
scripts/                # Lệnh sinh artifacts
src/darp/               # Dữ liệu, mô hình, đánh giá và dịch vụ quyết định
tests/                  # Kiểm thử và fixture hồi quy
ui/                     # Giao diện Streamlit và API client
Makefile                # Lệnh cài đặt, kiểm thử và chạy
pyproject.toml          # Metadata package và dependencies
requirements.txt        # Dependencies dạng requirements
```

## Cài đặt và chạy nhanh

Cần Python 3.10 trở lên. Cài đặt thường mất khoảng 1–3 phút; `make precompute` thường mất vài phút và phụ thuộc CPU, I/O và cấu hình máy.

```bash
python -m venv .venv
source .venv/bin/activate
make install
make precompute
```

Chạy API và giao diện trong hai terminal riêng:

```bash
make api
make ui
```

API: `http://localhost:8000` (tài liệu tại `/docs`). Giao diện: `http://localhost:8501`.

Các lệnh tương đương khi không có `make`:

```bash
python -m pip install -e ".[test]"
python -m scripts.precompute --config configs/default.yaml --out artifacts/
PYTHONPATH=. python -m uvicorn api.main:app --reload --port 8000
python -m streamlit run ui/app.py
```

## API

| Phương thức | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Kiểm tra trạng thái API |
| GET | `/meta` | Cấu hình và danh sách tháng, analyst, broker, ticker |
| GET | `/results` | Kết quả phân tích đã sinh |
| GET | `/events` | Danh sách sự kiện theo tháng; hỗ trợ lọc và giới hạn |
| GET | `/events/{event_id}` | Chi tiết quyết định và giải thích |
| POST | `/decide` | Tính quyết định cho danh sách khuyến nghị mới |
| GET | `/toy` | Kết quả ví dụ minh họa |
| POST | `/toy` | Tính quyết định cho dữ liệu minh họa tùy chỉnh |
| POST | `/experiment` | Chạy thực nghiệm với tham số chỉ định |

Lấy các giá trị hợp lệ từ `GET /meta`. Ví dụ sau sử dụng `A0000`, `S0000` và tháng `2017-02`, có trong artifacts sinh từ cấu hình mặc định:

```bash
curl -X POST http://localhost:8000/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "month": "2017-02",
    "ticker": "S0000",
    "recommendations": [
      {"analyst": "A0000", "rating": 4}
    ]
  }'
```

## Giao diện

- **Kết quả tái hiện**: ACC, khoảng tin cậy, phân tích độ nhạy và độ tin cậy.
- **Ra quyết định**: lọc sự kiện, xem kết quả và láng giềng ảnh hưởng.
- **Ví dụ minh họa**: thử ví dụ nhỏ và xem điểm tin cậy.
- **Chạy thực nghiệm**: chạy đánh giá với tham số người dùng chọn.

## Dữ liệu

IBES và CRSP thuộc các cơ sở dữ liệu WRDS và thường yêu cầu quyền truy cập; vì vậy cấu hình mặc định dùng dữ liệu mô phỏng với seed cố định. Bộ sinh tạo khuyến nghị, giá, chuyên gia, môi giới và đặc tính cổ phiếu; `true_q` là “sự thật ngầm” chỉ dùng để đánh giá tương quan sau khi học, không được dùng làm đầu vào quyết định. Có thể nạp dữ liệu thật qua `darp/data/loaders.py`: `load_ibes`, `load_crsp` và `load_stocks` chuyển dữ liệu chuẩn hóa về schema của package.

## Kết quả chính

Các giá trị dưới đây được sinh từ package với cấu hình mặc định; fixtures kiểm thử hồi quy được tạo từ cùng pipeline. ACC là tỷ lệ dự đoán đúng; CI là khoảng tin cậy bootstrap 95%.

| Chỉ số | MV | WM | DARP |
|---|---:|---:|---:|
| ACC | 0.6350 | 0.6562 | 0.6688 |
| CI 95% – cận dưới | 0.6259 | 0.6469 | 0.6597 |
| CI 95% – cận trên | 0.6452 | 0.6646 | 0.6767 |

### \(\eta\) và ablation

Giá trị \(\eta\) tham chiếu lấy từ bài báo; giá trị package lấy từ `eta_mean`.

| Đặc trưng | \(\eta\) tham chiếu | Package |
|---|---:|---:|
| f1 | 3.540 | 2.804 |
| f2 | 2.630 | 2.213 |
| f3 | 1.100 | 1.383 |
| f4 | 1.270 | 1.516 |
| f5 | 1.480 | 1.635 |
| f6 | 3.390 | 1.817 |

| Ablation | ACC |
|---|---:|
| DARP | 0.6688 |
| Bỏ f1 | 0.6576 |
| Bỏ f2 | 0.6538 |
| Bỏ f3 | 0.6745 |
| Bỏ f4 | 0.6752 |
| Bỏ f5 | 0.6701 |
| Bỏ f6 | 0.6665 |

**Khác biệt so với bài báo:** các giá trị \(\eta\) và ACC là kết quả tính trên bộ dữ liệu mô phỏng của repo; chúng không nhất thiết trùng số liệu công bố do thiết lập dữ liệu và một số lựa chọn cài đặt khác nhau. Fixtures hồi quy xác nhận tính nhất quán giữa các lần chạy package, không phải đối chiếu độc lập với dữ liệu bên ngoài.

## Ánh xạ bài báo → code

| Nội dung | Module / hàm |
|---|---|
| Eq. (5), độ liên quan có dấu \(\delta\) | `darp.relevance.build_graph` |
| Eq. (6), tối ưu độ tin cậy \(c\) | `darp.model.solve_c` |
| Eq. (7), tổng hợp dự đoán \(\rho\) | `darp.model.darp_fit` |
| Eq. (9), Theorem 1 | `darp.model.solve_c`; `tests/test_model.py` |
| Theorem 2, cập nhật \(\eta\) | `darp.model.update_eta`; `tests/test_model.py` |
| Algorithm 1 | `darp.model.darp_fit` |
| Table 2, ma trận chi phí | `darp.metrics.COST`, `darp.metrics.acc_cost` |
| Table 3, đặc trưng f1–f6 | `darp.relevance.pair_features` |
| Đánh giá theo tháng | `darp.evaluation.run_month`, `darp.evaluation.run_rolling` |
| Phân tích kết quả | `darp.analysis`; `scripts/precompute.py` |

## Kiểm thử

```bash
make test
make test-all
```

Các kiểm thử lý thuyết gồm Laplacian có dấu nửa xác định dương (Theorem 1), chuẩn hóa \(\sum \exp(-\eta)=1\), lịch sử hàm mục tiêu không tăng (Theorem 2 / Algorithm 1), và ví dụ minh họa với \(c(a_1),c(a_2),c(a_3)\), MV và DARP. `make test-all` còn chạy hồi quy chậm so với fixture.

## Lựa chọn cài đặt (bài báo không nêu) & hạn chế

- Đồ thị thưa giới hạn 30 láng giềng khác sự kiện cho mỗi nút.
- Jaccard dùng cửa sổ lịch sử 12 tháng.
- Tối ưu giới hạn 30 vòng lặp, `tol=1e-6`, và dùng ridge `1e-6` cho hệ tuyến tính.
- Phân tích độ nhạy mặc định chạy trên các tháng chẵn để giảm thời gian.
- Dữ liệu mô phỏng không thay thế dữ liệu IBES/CRSP; kết quả số phụ thuộc cấu hình mô phỏng.
- Bộ nạp dữ liệu thật yêu cầu dữ liệu đã được chuẩn hóa theo schema được mô tả ở phần Dữ liệu.

## Trích dẫn

```bibtex
@article{zhong2024decision,
  author  = {Zhong, H. and Chen, Y. and Liu, C. and Benson, H.},
  title   = {Decision aggregation with reliability propagation},
  journal = {Decision Support Systems},
  volume  = {178},
  pages   = {114130},
  year    = {2024},
  doi     = {10.1016/j.dss.2023.114130}
}
```

## Tác giả

Nguyễn Hoàng Phúc – 250101100 · Môn Hệ hỗ trợ quyết định (CS2208.11.CH) · Trường ĐH Công nghệ Thông tin – ĐHQG TP.HCM
