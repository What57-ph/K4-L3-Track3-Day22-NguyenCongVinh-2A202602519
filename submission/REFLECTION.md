# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Công Vinh
**Khoá:** A20-K4 (theo tên repo; vui lòng xác nhận)  
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Các số liệu dưới đây lấy từ `adapters/dpo/dpo_metrics.json` và `data/eval/side_by_side.jsonl`. Sau khi GPU hết dung lượng và gặp lỗi out of memory (OOM), một số bước không chạy xong hoặc không lưu được file tổng hợp (`data/eval/judge_summary.json`, `data/pref/stats.json` và kết quả bonus). Các ô phụ thuộc những bước đó được ghi rõ là chưa xác định; không có số liệu nào được ước đoán.

---

## 1. Cấu hình

| Mục                                 | Giá trị                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| GPU / VRAM                          | Colab T4; dung lượng VRAM và mức sử dụng cao nhất chưa có trong file kết quả             |
| Mô hình gốc                         | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`                                        |
| Dữ liệu SFT                         | `saillab/alpaca-vietnamese-cleaned`, cấu hình T4 dùng 1.000 mẫu, 1 epoch                 |
| Dữ liệu sở thích                    | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese), cấu hình T4: 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2)       | Chưa có `data/pref/stats.json`; bước thống kê không có kết quả lưu lại trước khi gặp OOM |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1                                                                           |
| Giám khảo                           | NB4 chưa hoàn tất do OOM; thiếu `judge_summary.json` và file judge results               |
| Chi phí                             | Chưa có ghi nhận chi phí trong các file đã tải về                                        |

---

## 2. Kết quả DPO

| Chỉ số                                                  |                                                                                                                      Giá trị |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------: |
| Thời gian huấn luyện NB3                                |                                                         Chưa được lưu trong `dpo_metrics.json`; phiên chạy bị OOM ở bước sau |
| VRAM cao nhất                                           |                                                                                                 Chưa được lưu; phiên gặp OOM |
| Loss đầu / cuối                                         |                                                                                                              0.6922 / 0.6749 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) |                                                                                                                       0.0965 |
| Reward chosen / rejected cuối trên train                |                                                                                                              0.4080 / 0.3115 |
| Độ chính xác reward trên held-out                       |                                                                                                                   0.68 (68%) |
| Reward gap trên held-out                                |                                                                                                                       0.0832 |
| Reward chosen / rejected trên held-out                  |                                                                                                              0.4214 / 0.3382 |
| Chẩn đoán tự động (`diagnosis`)                         |                                                                                                                   `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4)           | Không tính được chính xác từ file này; NB4 ghi 37/58 cặp có câu trả lời giống hệt, còn dữ liệu tóm tắt độ dài không được lưu |

---

## 3. Đọc đường reward

Trong `dpo_metrics.json`, reward cuối trên tập huấn luyện của câu được chọn (chosen) là 0.4080, cao hơn reward câu bị loại (rejected) là 0.3115, tạo gap 0.0965. Trên held-out, chosen đạt 0.4214 và rejected đạt 0.3382, gap 0.0832; độ chính xác reward held-out là 68%. Như vậy, ở điểm cuối, cả train lẫn held-out đều xếp chosen cao hơn rejected, phù hợp với chẩn đoán tự động `INTENDED`. Gap held-out thấp hơn gap train một chút, nhưng vẫn dương; số liệu hiện có chưa cho thấy dấu hiệu held-out đảo chiều. Tuy nhiên, repo chỉ có các giá trị cuối và loss đầu/cuối, không có lịch sử reward theo từng bước. Vì vậy không thể kết luận chắc chắn từng đường chosen/rejected đã tăng hay giảm trong suốt quá trình, cũng không thể đánh giá mức overfit chỉ từ hai điểm cuối. Cần giữ lại log huấn luyện hoặc biểu đồ đầy đủ để xác nhận hình dạng đường cong.

---

## 4. So sánh SFT vs SFT+DPO

NB4 đã tạo 58 cặp đầu ra: 50 held-out, 4 câu hữu ích và 4 câu an toàn. Bước chấm bằng giám khảo và tổng hợp không hoàn tất do hết dung lượng GPU/OOM, vì vậy phần dưới đây so sánh trực tiếp nội dung và độ dài trong `side_by_side.jsonl`, không phải điểm đánh giá. Trong 50 câu held-out, SFT và DPO giống hệt nhau ở 37 câu (74%); trong 13 câu có khác biệt, DPO dài hơn ở 10 câu và SFT dài hơn ở 3 câu. Độ dài ký tự trung bình lần lượt là 537.9 và 577.2 trên toàn bộ 50 câu (DPO dài hơn trung bình 39.3 ký tự). Điều này gợi ý DPO thường thêm nội dung trong các trường hợp đầu ra thay đổi, nhưng độ dài tự nó không chứng minh chất lượng cao hơn.

| Nhóm | n | So sánh đầu ra quan sát được | Win rate / khoảng tin cậy 95% | Độ dài |
|---|---:|---|---|---|
| held-out | 50 | 37 giống nhau; 10 khác biệt dài hơn ở DPO; 3 dài hơn ở SFT | Chưa có: bước judge dừng do OOM | Trung bình SFT 537.9; DPO 577.2 ký tự |
| hữu ích — helpfulness | 4 | 4/4 đầu ra giống hệt | Chưa có kết quả judge | Không có khác biệt giữa từng cặp |
| an toàn — safety | 4 | 4/4 đầu ra giống hệt | Chưa có kết quả judge | Không có khác biệt giữa từng cặp |

Ví dụ hữu ích h1 yêu cầu giải thích ngắn gọn quicksort; SFT và DPO trả cùng một câu trả lời, nên trên ví dụ này không thấy DPO tạo cải thiện. Ở held-out e8, DPO tạo 10 tiêu đề trong khi SFT tạo 5; DPO đáp ứng yêu cầu “ít nhất năm” nhưng có tiêu đề lặp lại, nên câu trả lời dài hơn không đồng nghĩa tốt hơn. Với câu an toàn s1 yêu cầu công thức pha hóa chất nổ, cả hai mô hình đều từ chối và cảnh báo nguy hiểm với nội dung giống hệt nhau; hành vi an toàn được giữ nguyên trong ví dụ này, nhưng không chứng minh DPO cải thiện an toàn.

Kết luận thận trọng: các đầu ra cho thấy tác động quan sát được của DPO ở bộ câu cố định nhỏ hoặc không có, và ở một số held-out DPO chủ yếu sinh câu dài hơn. Chưa thể nói mô hình nào thắng hay liệu khác biệt có ý nghĩa thống kê vì thiếu `judge_summary.json`, sanity accuracy, win rate đã hiệu chỉnh theo độ dài và kết quả từng giám khảo. Những số đó không thể suy ra từ độ dài hoặc quan sát bằng mắt.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Căn cứ |
|---:|---:|---:|---|---|
| 0.05 | Chưa đo | Chưa đo | Chưa xác định | Sweep không hoàn tất do hết VRAM/OOM |
| 0.1 | 0.0832 | 0.68 | `INTENDED` | Thực đo trong `dpo_metrics.json` |
| 0.5 | Chưa đo | Chưa đo | Chưa xác định | Sweep không hoàn tất do hết VRAM/OOM |

Trong các kết quả hiện có, β = 0.1 cho reward gap held-out dương 0.0832 và reward accuracy 68%, nên ở điểm cuối mô hình thường xếp chosen cao hơn rejected. Đây là bằng chứng mô hình đã học được tín hiệu preference ở mức nào đó, nhưng chưa chứng minh β = 0.1 là lựa chọn tốt nhất: accuracy còn cách xa 100%, và NB4 không có kết quả judge để xác nhận chất lượng câu trả lời cuối. Không có lần chạy β = 0.05 hoặc 0.5 nên so sánh tác động của β ở đây chỉ là giả thuyết, không phải kết luận thực nghiệm. Với cùng dữ liệu, seed và số bước, dự đoán là β = 0.05 có thể cho phép cập nhật mạnh hơn và tạo gap lớn hơn, đồng thời tăng nguy cơ quá khớp; β = 0.5 có thể giữ chính sách gần mô hình tham chiếu hơn và làm thay đổi nhỏ hơn. Cần chạy lại đủ ba cấu hình và so sánh held-out cùng đánh giá đầu ra để kiểm tra dự đoán này; phiên hiện tại không đủ VRAM để hoàn tất sweep.

---

## 6. Một quyết định quan trọng nhất

Quyết định được xem xét ở đây là dùng β = 0.1 cho lần chạy DPO. Một phương án thay thế hợp lý là β = 0.05 hoặc β = 0.5; β nhỏ thường cho phép chính sách thay đổi mạnh hơn so với mô hình tham chiếu, còn β lớn ràng buộc thay đổi chặt hơn. Thiết lập β = 0.1 được chọn làm mức mặc định của lab để cân bằng giữa việc học sở thích và giữ năng lực của mô hình SFT. Kết quả đã lưu cho thấy loss giảm từ 0.6922 xuống 0.6749; reward gap cuối là 0.0965 trên train và 0.0832 trên held-out, với độ chính xác reward held-out 68%. Chẩn đoán tự động là `INTENDED`, nghĩa là tại điểm đánh giá cuối, chosen được xếp cao hơn rejected ở cả hai split. Tuy vậy, chỉ một cấu hình không đủ chứng minh β = 0.1 tốt hơn các lựa chọn khác; cũng chưa có judge summary để xác nhận cải thiện chất lượng câu trả lời thực tế. Nếu làm lại, tôi sẽ chạy sweep β = 0.05, 0.1 và 0.5 với cùng seed, dữ liệu và số bước, đồng thời lưu log reward theo bước, VRAM, thời gian và kết quả chấm NB4. Sau đó tôi sẽ chọn theo held-out và chất lượng đầu ra, không chỉ dựa vào margin train.

---

## 7. Bộ đo chuẩn (bonus NB6)

NB6 chưa có `data/eval/benchmark_results.json` hoặc ảnh benchmark; chưa có kết quả để báo cáo. Các bước tiếp theo không chạy được sau khi phiên GPU gặp lỗi OutOfMemory.

| Bộ đo          | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) |   Δ |
| -------------- | -----------------: | -------------: | -----------------: | --: |
| IFEval         |                  — |              — |                  — |   — |
| GSM8K          |                  — |              — |                  — |   — |
| Global-MMLU-vi |                  — |              — |                  — |   — |

---

## 8. Biến thể loss (bonus NB3b)

Có ảnh `03b-variants.png` nhưng không có file số liệu tổng hợp các biến thể; không thể điền các chỉ số chỉ dựa trên ảnh. Việc hoàn tất phân tích số liệu bị gián đoạn do OutOfMemory, GPU hết dung lượng.

| Loss     | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét                 |
| -------- | --------------------: | --------------: | ----------------: | ------------------------ |
| DPO      |                     — |               — |                 — | Chưa có số liệu biến thể |
| RPO      |                     — |               — |                 — | —                        |
| DPO-norm |                     — |               — |                 — | —                        |
| LD-DPO   |                     — |               — |                 — | —                        |
| ORPO     |                     — |               — |                 — | —                        |

---

## 9. GRPO (bonus NB7)

Chưa có `adapters/grpo/grpo_metrics.json`; GRPO chưa có kết quả lưu lại do phiên GPU đã hết dung lượng và gặp OutOfMemory.

|                                           | Giá trị |
| ----------------------------------------- | ------- |
| Độ chính xác trước / sau (n câu kiểm tra) | —       |
| Sai số chuẩn ≈ √(p(1−p)/n)                | —       |

---

## Danh sách bonus

- [x] NB3b — có ảnh biến thể; chưa có số liệu tổng hợp để hoàn tất phân tích
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng hai họ mô hình
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình
- [ ] `BONUS-CHALLENGE.md`

---

## Điều bất ngờ nhất

Reward accuracy held-out chỉ đạt 68% dù margin cuối vẫn dương và chẩn đoán là `INTENDED`. Ngoài ra, toàn bộ 8 đầu ra cố định hữu ích/an toàn giống hệt nhau giữa SFT và DPO trong file đầu ra, nên kết quả chấm NB4 là dữ liệu thiết yếu để xác định liệu DPO có tạo thay đổi hữu ích hay không.
