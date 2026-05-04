# Hướng Dẫn Sử Dụng - FL Comparison Files

## Tổng quan

Đã tạo **2 files riêng** để so sánh FL với các baseline khác nhau:

### 1. **FL_vs_Centralized.py**
So sánh FL với Centralized FNN training (bao gồm Domain-Specific Analysis)
- **Focus**: Efficiency, Per-IBR Fairness (vs Centralized), Impedance physics check
- **Thời gian chạy**: ~30-60 seconds (do phải run prediction trên test sets)
- **Visualizations**: 5 Figures lớn (Convergence, Final Metrics, Per-IBR, Impedance Bode, Error Dist)

### 2. **FL_vs_LocalOnly.py**  
So sánh FL với Local-Only training (mỗi client train riêng)
- **Focus**: Per-client fairness, variance reduction, individual improvements
- **Thời gian chạy**: ~5-10 phút (phải train 9 local models)
- **Visualizations**: 10 plots trong 2 figures

---

## Cách Sử Dụng

### Bước 1: Chuẩn bị (chạy các cells trong notebook TRƯỚC)

Trong `FL_SCENARIO_AE.ipynb`, chạy theo thứ tự:
1. ✅ **Cell 1-5**: Load data, scaling
2. ✅ **Cell 6**: Model definitions (`HIDDEN_GFLI`, `Trunk`, `Head`, `FullModel`)
3. ✅ **Cell 7**: FedAvg training → tạo `fullfedavg_model_state`, `history_fullfedavg`
4. ✅ **Cell 9**: Central FNN training → tạo `central_train_curve`, `central_test_curve`, **`model`** (central model)

### Bước 2: Chạy file so sánh

**Option A - Tách biệt**:
```python
# Cell mới: So sánh FL vs Centralized (Advanced)
%run FL_vs_Centralized.py
```

```python
# Cell mới khác: So sánh FL vs Local-Only  
%run FL_vs_LocalOnly.py
```

**Option B - Chạy cả hai**:
```python
# Chạy cả 2 files liên tiếp
%run FL_vs_Centralized.py
%run FL_vs_LocalOnly.py
```

---

## Chi Tiết Từng File

### 📊 FL_vs_Centralized.py (MỚI CẬP NHẬT)

**Figure 1: Performance Comparison (2x2 grid)**
- Learning curves & Convergence analysis

**Figure 2: Final Metrics (1x3 grid)**
- Final MSE, AUC, Generalization gap

**Figure 3: Per-IBR Performance (NEW)**
- Bar chart so sánh MSE của FL vs Centralized trên từng khách hàng.
- Hiển thị xem khách hàng nào bị "thiệt" khi dùng FL so với Centralized.

**Figure 4: Domain-Specific Impedance (NEW)**
- **Z_true vs Z_pred**: Plot giá trị đầu ra (8 output dims) cho 1-2 mẫu đại diện (Bode-style).
- So sánh khả năng bắt đúng đặc tính vật lý của FL vs Centralized.

**Figure 5: Error Distribution (NEW)**
- **Histogram & CDF**: Phân phối lỗi của toàn hệ thống (MAPE).
- Chứng minh FL không có "đuôi" lỗi lớn (worst-case error).

**Output Metrics**:
- Convergence speed, AUC
- Per-IBR MSE comparison
- System-wide Mean MAPE
- 90th percentile Error

---

### 👥 FL_vs_LocalOnly.py

**Figure 1: Per-Client Analysis (2x2 grid)**
1. Bar chart: MSE comparison cho từng client
2. Horizontal bar: % improvement cho mỗi client
3. Scatter plot: Sample size vs improvement (tìm pattern)
4. Box plot: Distribution comparison

**Figure 2: Fairness Analysis (1x2 grid)**
1. Normalized performance với arrows showing improvements
2. Individual & cumulative improvements (sorted)

**Output Metrics**:
- Average improvement %
- Variance reduction %
- Clients improved (X/9)
- Fairness improvement (performance range)
- Best/worst performing clients

---

## Kết Quả Mong Đợi

### FL vs Centralized:
✅ Comparable or better final performance  
✅ Similar or faster convergence  
✅ Better generalization (smaller gap)  
✅ More efficient learning (lower AUC)

### FL vs Local-Only:
✅ Better performance for most clients (especially small-data clients)  
✅ Reduced variance across clients  
✅ Improved fairness (reduced performance gap)  
✅ 70-90% of clients show improvement

---

## Troubleshooting

### Lỗi: "NameError: name 'HIDDEN_GFLI' is not defined"
→ Chưa chạy Cell 6 trong notebook. Chạy lại Cell 1-9 trước.

### Lỗi: "NameError: name 'fullfedavg_model_state' is not defined"  
→ Chưa chạy FedAvg training (Cell 7). Chạy lại.

### Lỗi: "NameError: name 'central_train_curve' is not defined"
→ Chưa chạy Central FNN training (Cell 9). Chạy lại.

### CUDA Out of Memory (khi chạy FL_vs_LocalOnly.py)
→ Giảm batch_size hoặc train trên CPU:
```python
# Thêm vào đầu file
device = torch.device('cpu')
```

---

## Customization

### Thay đổi convergence targets:
```python
# Trong FL_vs_Centralized.py, line ~28
convergence_targets = [0.85, 0.90, 0.95]  # Thay đổi các mốc
```

### Thay đổi epochs cho local training:
```python
# Trong FL_vs_LocalOnly.py, line ~67
local_mse = train_local_only_model(client, test_sets_gfli[i], epochs=50)
```

### Thay đổi màu sắc plots:
- `tab:blue` → FedAvg  
- `tab:orange` → Centralized
- `tab:red` → Local-Only

---

## So Sánh 2 Files

| Khía cạnh | FL_vs_Centralized | FL_vs_LocalOnly |
|-----------|-------------------|-----------------|
| **Thời gian chạy** | ~15s | ~5-10 phút |
| **Số figures** | 2 | 2 |
| **Số plots** | 7 | 10 |
| **Focus** | Global efficiency | Per-client fairness |
| **Metrics chính** | AUC, Convergence | Variance, Improvement% |
| **Khó khăn** | Không | Train 9 models |

---

## Lưu Ý Quan Trọng

1. **Dependencies**: Cần các biến từ notebook, KHÔNG thể chạy standalone
2. **Thứ tự**: Phải chạy notebook cells trước khi chạy các files này
3. **Memory**: FL_vs_LocalOnly cần nhiều memory hơn
4. **Output**: Cả 2 files đều print detailed statistics ra console

---

**File gốc cũ `FL_visualization_cells.py` đã KHÔNG còn được khuyến nghị sử dụng.**  
Sử dụng 2 files mới này để phân tích rõ ràng hơn!
