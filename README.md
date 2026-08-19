# 📰 Fake News Detection using Machine Learning & mBERT

## 📌 GIỚI THIỆU ĐỀ TÀI
--
Sự phát triển mạnh mẽ của Internet và các nền tảng mạng xã hội đã tạo điều kiện cho thông tin được lan truyền với tốc độ chưa từng có. Tuy nhiên, đi cùng với đó là sự gia tăng của **tin tức giả (Fake News)**, có khả năng tác động tiêu cực đến nhận thức cộng đồng và các quyết định trong đời sống xã hội.
Đồ án này tập trung xây dựng một **hệ thống tự động phát hiện và phân loại tin tức giả** dựa trên nội dung văn bản. Nhóm thực hiện so sánh hai hướng tiếp cận chính:

- **Traditional Machine Learning:** sử dụng **TF-IDF** để biểu diễn văn bản, sau đó huấn luyện nhiều mô hình phân loại.
- **Deep Learning:** sử dụng **Multilingual BERT (mBERT)** và fine-tuning cho bài toán phân loại Fake News.

Mục tiêu của đồ án không chỉ là tìm ra mô hình có hiệu năng cao mà còn tiến hành **phân tích lỗi, kiểm tra shortcut learning và potential data leakage**, từ đó đánh giá mức độ tin cậy của kết quả thực nghiệm.

###🎯 MỤC TIÊU 
| # | Mục tiêu |
|:---:|:---|
| **01** | Khảo sát và phân tích bộ dữ liệu **WELFake** |
| **02** | Xây dựng quy trình xử lý dữ liệu và huấn luyện hình hoàn chỉnh cho bài toán phân loại tin giả |
| **03** | Đánh giá và so sánh hiệu quả của các mô hình thông qua các chỉ số thực nghiệm |
| **04** | Khảo sát nguy cơ rò rỉ dữ liệu và các đặc trưng tiềm ẩn có thể ảnh hưởng đến kết quả |
| **05** | Đề xuất các hướng phát triển cho hệ thống |

## 📊 BỘ DỮ LIỆU 
--
Đồ án sử dụng bộ dữ liệu **WELFake**, một bộ dữ liệu dành cho bài toán Fake News Detection được tổng hợp từ nhiều nguồn dữ liệu khác nhau.
| Thành phần | Thông tin |
|:---|:---|
| **Tên bộ dữ liệu** | WELFake |
| **Ngôn ngữ** | tiếng Anh |
| **Số lượng mẫu** | **72,134 bài báo** |
| **Đầu vào** | `Title + Text` |
| **Nhãn** | `0 = Fake News` · `1 = Real News` |
| **Bài toán** | Phân loại văn bản |

### 📈 Phân bố dữ liệu
| Lớp | Nhãn | Số lượng mẫu |
|:---|:---:|---:|
| 🚨 **Fake News** | `0` | **37,106** |
| 📰 **Real News** | `1` | **35,028** |
| **Total** | — | **72,134** |

Bộ dữ liệu ban đầu có phân bố tương đối cân bằng giữa hai lớp Fake News và Real News.

## 🧹 QUY TRÌNH THỰC NGHIỆM
--
![Quy trình thực nghiệm phát hiện tin giả]()

Sau quá trình làm sạch, số lượng mẫu giảm từ **72,134 → 63,279**.
| Giai đoạn xử lý | Số lượng mẫu còn lại | Số lượng mẫu đã loại bỏ |
|:---|---:|---:|
| Dữ liệu gốc | 72,134 | — |
| Giá trị khuyết | 72,095 | 39 |
| Các bài báo ngắn (dưới 5 từ) | 72,010 | 85 |
| Khử trùng lặp gần đúng | **63,279** | **8,731** |

## 🤖 MÔ HÌNH THỰC NGHIỆM 
--
### Mô hình học máy truyền thống 
Các mô hình học máy truyền thống sử dụng **TF-IDF** để chuyển đổi văn bản thành vector đặc trưng. 
| Phân loại | Các mô hình |
|:---|:---|
| **Linear Models** | Logistic Regression |
| | Linear SVM |
| | SGD Classifier |
| | Ridge Classifier |
| | Passive Aggressive |
| **Naive Bayes** | Multinomial NB |
| | Complement NB |
| | Bernoulli NB |
| **Ensemble** | Random Forest |
| | XGBoost |

### mBERT 
**Multilingual BERT (mBERT)** được sử dụng theo phương pháp **Fine-tuning** cho bài toán phân loại văn bản nhị phân.
Thay vì biểu diễn văn bản dựa chủ yếu trên tần suất xuất hiện của từ như TF-IDF, mBERT sử dụng **Transformer Architecture và Self-Attention** để xây dựng biểu diễn ngữ nghĩa theo ngữ cảnh.
| Siêu tham số | Giá trị |
|:---|:---:|
| **Learning Rate** | `2e-5` |
| **Train Batch Size** | `16` |
| **Evaluation Batch Size** | `16` |
| **Epochs** | `3` |
| **Weight Decay** | `0.01` |
| **Warmup Ratio** | `0.1` |
| **Best Model Metric** | `F1-Macro` |

## 📈 KẾT QUẢ THỰC NGHIỆM 
-- 
Các mô hình được đánh giá dựa trên:

| Chỉ số đánh giá | Mô tả |
|:---|:---|
| **Accuracy** | Tỷ lệ dự đoán đúng trên toàn bộ mẫu |
| **Precision** | Độ chính xác của các dự đoán dương |
| **Recall** | Khả năng phát hiện đúng các mẫu thuộc lớp |
| **Macro F1** | F1 trung bình giữa hai lớp |
| **Fake F1** | F1-score riêng cho lớp Fake News |

### Baseline - Các mô hình học máy truyền thống 
| Rank | Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Fake F1 | Train Time (s) |
|:---:|:---|---:|---:|---:|---:|---:|---:|
| 🥇 | **Linear SVM** | **0.9742** | **0.9740** | **0.9742** | **0.9739** | **0.9764** | 1.73 |
| 🥈 | Passive Aggressive | 0.9723 | 0.9721 | 0.9722 | 0.9720 | 0.9747 | 1.05 |
| 🥉 | Ridge Classifier | 0.9722 | 0.9720 | 0.9719 | 0.9720 | 0.9745 | 4.36 |
| 4 | XGBoost | 0.9685 | 0.9682 | 0.9679 | 0.9686 | 0.9710 | 316.70 |
| 5 | SGD (hinge) | 0.9659 | 0.9656 | 0.9655 | 0.9657 | 0.9687 | 0.63 |
| 6 | Logistic Regression | 0.9610 | 0.9606 | 0.9608 | 0.9605 | 0.9643 | 1.72 |
| 7 | Random Forest | 0.9302 | 0.9295 | 0.9301 | 0.9289 | 0.9364 | 117.70 |
| 8 | Complement NB | 0.8756 | 0.8751 | 0.8746 | 0.8775 | 0.8825 | 0.19 |
| 9 | Multinomial NB | 0.8752 | 0.8746 | 0.8739 | 0.8763 | 0.8832 | 0.20 |
| 10 | Bernoulli NB | 0.8499 | 0.8498 | 0.8535 | 0.8554 | 0.8525 | 0.39 |

Linear SVM đạt hiệu năng cao nhất trong nhóm Machine Learning truyền thống. Kết quả này cho thấy các mô hình tuyến tính hoạt động hiệu quả trong không gian đặc trưng **TF-IDF** có số chiều lớn và thưa.
 ### mBERT 
| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Fake F1 | Train Time |
|:---:|---:|---:|---:|---:|---:|---:|
| 🏆 **mBERT** | **99.57%** | **99.57%** | **99.58%** | **99.56%** | **99.61%** | 3778.33s |

### ⚔️ Baseline vs. mBERT
| Hướng | Mô hình tốt nhất | Accuracy | Macro F1 |
|:---|:---|---:|---:|
| Mô hình truyền thống | Linear SVM | 97.42% | 97.40% |
| **Transformer** | **mBERT** | **99.57%** | **99.57%** |
| **Cải thiện** | — | **+2.15 pp** | **+2.17 pp** |
> 💡 **Kết luận:** TF-IDF + Linear SVM đã tạo ra một baseline rất mạnh. Tuy nhiên, mBERT vẫn đạt hiệu năng cao hơn nhờ khả năng học **trình bày ngữ nghĩa** và khai thác quan hệ ngữ nghĩa trong văn bản.

## 🔍 PHÂN TÍCH LỖI 
--
Mặc dù mô hình đạt Accuracy rất cao, kết quả cần được kiểm tra để xác định liệu mô hình thực sự học được **nội dung và ngữ nghĩa của tin tức** hay đang tận dụng những đặc trưng bề mặt của dataset.

Phân tích dữ liệu cho thấy một số đặc trưng liên quan đến nguồn tin có phân bố rất khác nhau giữa hai lớp.

![Shortcut phân bố lệch giữa 2 lớp]()

## 🌍 HƯỚNG PHÁT TRIỂN TRONG TƯƠNG LAI 
--
- Việc mở rộng sang các bộ dữ liệu đa ngôn ngữ hoặc dữ liệu tiếng Việt sẽ giúp đánh giá tốt hơn khả năng thích nghi của mô hình trong các bối cảnh khác nhau.
- Khai thác thêm các nguồn thông tin ngoài nội dung văn bản
- Các mô hình Transformer thế hệ mới như RoBERTa, DeBERTa hoặc các mô hình ngôn ngữ lớn (Large Language Models - LLMs) đang cho thấy hiệu quả vượt trội trong nhiều bài toán xử lý ngôn ngữ tự nhiên. Việc nghiên cứu và áp dụng các mô hình này cho bài toán phát hiện tin giả là một hướng đi đầy triển vọng nhằm nâng cao độ chính xác và khả năng suy luận của hệ thống.
- Việc tích hợp các phương pháp giải thích mô hình sẽ giúp tăng tính minh bạch và mức độ tin cậy của hệ thống

## 📌 KẾT LUẬN
Nhóm đã thực hiện thành công việc xây dựng và đánh giá toàn diện hệ thống 
phân loại tin giả tự động dựa trên tập dữ liệu lớn WELFake. Quy trình tiền xử lý văn 
bản thiết kế riêng cho BERT đã chứng minh tính đúng đắn khi bảo toàn được các tín 
hiệu ngữ nghĩa quan trọng từ chữ viết hoa và dấu câu, kết hợp hiệu quả với thuật toán 
khử trùng lặp gần đúng để bảo vệ tính khách quan cho dữ liệu thực nghiệm. 
Kết quả thực chứng khẳng định mô hình học sâu mBERT đạt hiệu năng tối ưu 
tuyệt đối với độ chính xác 99.18% và điểm F1-macro 0.9917, vượt trội hoàn toàn so 
với thuật toán tốt nhất của trường phái học máy truyền thống là Linear SVM (97.42%). 
Mặc dù thực nghiệm đối chứng đã chỉ ra sự tồn tại của các "đường tắt" hệ thống dựa 
trên tên nguồn tin làm ảnh hưởng nhẹ đến điểm số thực tế, mô hình vẫn bảo toàn được 
năng lực phân loại ngữ nghĩa cốt lõi. Nghiên cứu này đặt nền móng vững chắc cho 
việc triển khai các ứng dụng bộ lọc tin tức tự động quy mô lớn trên thực tế nhằm bảo 
vệ không gian mạng an toàn. 










