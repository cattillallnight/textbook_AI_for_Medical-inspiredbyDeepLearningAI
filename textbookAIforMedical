# AI FOR MEDICINE SPECIALIZATION — TOÀN THƯ KIẾN THỨC CHUYÊN SÂU

> **Biên soạn dựa trên chương trình chuẩn của DeepLearning.AI**
>
> Bao gồm 3 học phần: **AI for Medical Diagnosis**, **AI for Medical Prognosis**, và **AI for Medical Treatment**.

# HỌC PHẦN 1: AI FOR MEDICAL DIAGNOSIS (CHẨN ĐOÁN HÌNH ẢNH Y KHOA)

Tập trung vào phân loại tổn thương (Classification) và khoanh vùng tổn thương (Segmentation) trên các phương tiện chẩn đoán hình ảnh: X-quang, CT-scan, và MRI.

## 1. Phân loại bệnh lý trên X-quang lồng ngực & Thách thức dữ liệu

### 1.1. Hiện tượng mất cân bằng lớp cực đoan (Extreme Class Imbalance)

* **Bản chất lâm sàng:** Trong các chiến dịch tầm soát sức khỏe cộng đồng, tỷ lệ người khỏe mạnh chiếm đa số tuyệt đối (khoảng $98\% - 99\%$), chỉ có $1\% - 2\%$ xuất hiện bệnh lý thực sự (viêm phổi, tràn dịch màng phổi, xẹp phổi).

* **Vấn đề kỹ thuật:** Nếu huấn luyện mạng nơ-ron với hàm mất mát Cross-Entropy tiêu chuẩn, mô hình sẽ hội tụ về điểm tối ưu cục bộ: luôn dự đoán mọi ca đều là "Bình thường" (Không bệnh). Khi đó:

  * Accuracy (Độ chính xác) đạt tới $99\%$.

  * Recall / Sensitivity (Độ nhạy) phát hiện bệnh bằng $0\%$. Toàn bộ người bệnh bị bỏ sót trong thực tế.

* **Giải pháp 1: Weighted Cross-Entropy Loss (Hàm mất mát có trọng số)**

  Gán trọng số phạt nặng hơn đáng kể khi mô hình dự đoán sai trên ca dương tính ($y=1$).

  $$
  L(y, \hat{y}) = - w_p \cdot y \log(\hat{y}) - w_n \cdot (1 - y) \log(1 - \hat{y})
  $$

  Quy tắc cân bằng tần suất:
  

  $$
  w_p = \frac{N_{neg}}{N_{pos} + N_{neg}}, \quad w_n = \frac{N_{pos}}{N_{pos} + N_{neg}}
  $$

  * *Ví dụ:* Tập dữ liệu có $1,000$ ảnh X-quang, trong đó chỉ có $20$ ca viêm phổi ($N_{pos} = 20$) và $980$ ca bình thường ($N_{neg} = 980$).

    * Trọng số dương tính: $w_p = \frac{980}{1000} = 0.98$

    * Trọng số âm tính: $w_n = \frac{20}{1000} = 0.02$

    * Khi mô hình đoán sai một ca bệnh ($y=1, \hat{y} \to 0$), mức phạt nặng gấp $\frac{0.98}{0.02} = 49$ lần so với khi đoán sai một ca lành.

* **Giải pháp 2: Resampling Kỹ thuật số**

  * **Oversampling:** Nhân bản các mẫu dương tính hiếm gặp trong từng batch huấn luyện.

  * **Undersampling:** Bỏ bớt ngẫu nhiên các mẫu âm tính để cân bằng tỷ lệ mẫu đưa vào mạng.

### 1.2. Phân loại đa nhãn (Multi-label Classification)

* **Phân biệt:**

  * **Multi-class:** Một ảnh chỉ có duy nhất 1 nhãn (dùng kích hoạt `Softmax`, tổng xác suất các lớp bằng $1.0$).

  * **Multi-label:** Một bệnh nhân có thể đồng thời mắc nhiều bệnh: vừa Tim to (*Cardiomegaly*), vừa Tràn dịch màng phổi (*Effusion*), vừa Viêm phổi (*Pneumonia*).

* **Cơ chế mạng:**

  * Lớp cuối cùng gồm $K$ nơ-ron độc lập, mỗi nơ-ron đi qua hàm **Sigmoid**:
    

    $$
    \hat{y}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \in [0, 1]
    $$

  * Hàm mất mát toàn thể là tổng Binary Cross-Entropy trên tất cả $K$ bệnh:
    

    $$
    L_{total} = \sum_{k=1}^K L_{BCE}(y_k, \hat{y}_k)
    $$

### 1.3. Cạm bẫy rò rỉ dữ liệu (Data Leakage via Patient ID)

* **Cơ chế gây lỗi:** Một bệnh nhân thường được chụp nhiều lần ở các tư thế khác nhau (PA, AP, Lateral) hoặc chụp theo dõi tiến triển bệnh qua nhiều tuần.

* **Hậu quả nếu chia ngẫu nhiên (Random Train-Test Split):** Ảnh của cùng một bệnh nhân xuất hiện ở cả tập Train và Test. Mạng CNN sẽ nhận diện hình dáng xương lồng ngực, bóng tim đặc thù hoặc phụ kiện kim loại cá nhân thay vì học bản chất bệnh học của tổn thương.

* **Quy tắc bắt buộc:** **Patient-level Split (Group K-Fold theo Patient ID)**. Mọi hình ảnh của cùng một bệnh nhân phải nằm hoàn toàn trong tập Train hoặc tập Test.

## 2. Đo lường & Đánh giá mô hình chẩn đoán

### 2.1. Ma trận nhầm lẫn & Các chỉ số lâm sàng

| 

| **Thực tế \\ Dự đoán** | **Dự đoán Dương tính (Y^=1)** | **Dự đoán Âm tính (Y^=0)** | 
| **Dương tính (**$Y=1$**)** | **True Positive (TP)** | **False Negative (FN)** *(Bỏ sót bệnh - Nguy cơ tử vong)* | 
| **Âm tính (**$Y=0$**)** | **False Positive (FP)** *(Báo động giả - Gây stress, chi phí)* | **True Negative (TN)** | 

* **Sensitivity (Độ nhạy / Recall / True Positive Rate):** 

  $$
  \text{Sensitivity} = \frac{TP}{TP + FN}
  $$

   *Khả năng không bỏ sót bất kỳ ai đang mang bệnh.*

* **Specificity (Độ đặc hiệu / True Negative Rate):** 

  $$
  \text{Specificity} = \frac{TN}{TN + FP}
  $$

   *Khả năng xác nhận chính xác một người thực sự khỏe mạnh.*

* **PPV (Positive Predictive Value / Precision):** 

  $$
  \text{PPV} = \frac{TP}{TP + FP}
  $$

   *Khi mô hình đưa ra cảnh báo có bệnh, xác suất người này thực sự mắc bệnh là bao nhiêu.*

* **NPV (Negative Predictive Value):** 

  $$
  \text{NPV} = \frac{TN}{TN + FN}
  $$

   *Khi mô hình kết luận an toàn, xác suất người này thực sự khỏe mạnh là bao nhiêu.*

### 2.2. Nghịch lý Prevalence (Tỷ lệ lưu hành) & Định lý Bayes

* **Đặc tính:** Sensitivity và Specificity là chỉ số kỹ thuật nội tại của thuật toán, độc lập với tỷ lệ dân số mang bệnh. Ngược lại, **PPV và NPV thay đổi hoàn toàn theo tỷ lệ lưu hành (Prevalence** $P$**)**:

  $$
  \text{PPV} = \frac{\text{Sensitivity} \times P}{\text{Sensitivity} \times P + (1 - \text{Specificity}) \times (1 - P)}
  $$

* **Ví dụ thực tế:** Một mô hình chẩn đoán ung thư gan có:

  * $\text{Sensitivity} = 99\%$ ($0.99$)

  * $\text{Specificity} = 99\%$ ($0.99$)

  * Áp dụng tầm soát tại địa phương có tỷ lệ lưu hành $P = 0.1\%$ ($0.001$).

  $$
  \text{PPV} = \frac{0.99 \times 0.001}{(0.99 \times 0.001) + (1 - 0.99) \times (1 - 0.001)} = \frac{0.00099}{0.00099 + 0.00999} \approx 9.01\%
  $$

  *Kết luận lâm sàng:* Mặc dù mô hình đạt độ nhạy và đặc hiệu $99\%$, nhưng khi máy báo "Dương tính", có tới gần $91\%$ khả năng bệnh nhân bị chẩn đoán nhầm (dương tính giả).

### 2.3. Đường cong ROC, PR & Kỹ thuật Bootstrap

* **Đường cong ROC (Receiver Operating Characteristic):** Vẽ quan hệ giữa $y = \text{Sensitivity}$ và $x = 1 - \text{Specificity}$ khi thay đổi ngưỡng phân loại $\tau \in [0, 1]$. Diện tích dưới đường cong là **AUROC**.

* **Đường cong PR (Precision-Recall):** Được ưu tiên khi tập dữ liệu mất cân bằng nghiêm trọng. AUROC có thể cao ảo do lượng lớn mẫu True Negative kéo False Positive Rate xuống thấp; PR Curve phản ánh trung thực sự sụt giảm của Precision.

* **Non-parametric Bootstrap ước tính khoảng tin cậy 95% (95% CI):**

  1. Cho tập Test $D$ kích thước $N$.

  2. Lấy mẫu ngẫu nhiên có hoàn lại (Sampling with replacement) $N$ phần tử từ $D$ để tạo tập con $D^*_b$.

  3. Tính metric (ví dụ AUROC) trên $D^*_b$.

  4. Lặp lại bước 2-3 trong $B = 1,000$ lần.

  5. Khoảng tin cậy $95\%$ là đoạn giữa phân vị thứ $2.5\%$ và $97.5\%$ của danh sách metric thu được.

## 3. Phân vùng ảnh y khoa (Medical Image Segmentation)

### 3.1. Kiến trúc U-Net (2D và 3D)

* **Cấu trúc chữ U đối xứng:**

  * **Encoder (Contracting Path):** Rút trích các đặc trưng ngữ nghĩa mức cao qua các lớp tích chập và giảm kích thước không gian bằng Max-Pooling.

  * **Decoder (Expanding Path):** Khôi phục kích thước không gian chi tiết bằng Transposed Convolutions (Upsampling).

* **Skip Connections (Cầu nối trực tiếp):** Nối trực tiếp bản đồ đặc trưng chi tiết từ tầng tương ứng của Encoder sang Decoder. Giúp Decoder giữ lại các ranh giới mô giải phẫu sắc nét vốn bị mờ đi khi giảm độ phân giải.

* **3D U-Net:** Áp dụng phép tích chập không gian 3 chiều ($3 \times 3 \times 3$) trên dữ liệu khối thể tích (CT hoặc MRI 3D) để khai thác tính liên tục giữa các lát cắt lân cận.

### 3.2. Hàm mất mát phân vùng: Soft Dice Loss

* Phân vùng khối u thường có tỷ lệ pixel khối u cực nhỏ so với pixel mô lành (background). Hàm Cross-Entropy từng điểm ảnh sẽ bị nền chi phối hoàn toàn.

* **Hệ số Dice (Sørensen–Dice Coefficient):** Đo độ trùng lặp giữa mặt nạ nhãn thực tế $P$ và vùng dự đoán $Q$: 

  $$
  \text{Dice}(P, Q) = \frac{2 \vert{}P \cap Q\vert{}}{\vert{}P\vert{} + \vert{}Q\vert{}} = \frac{2 \sum_i p_i q_i}{\sum_i p_i^2 + \sum_i q_i^2}
  $$

* **Soft Dice Loss:** 

  $$
  L_{\text{Dice}} = 1 - \text{Dice}(P, Q)
  $$

   Đạo hàm trơn tru cho phép tối ưu trực tiếp bằng phương pháp Gradient Descent, không phụ thuộc vào kích thước vùng nền.

### 3.3. Sub-volume Patch Sampling

* Một khối ảnh MRI sọ não 3D có thể có kích thước $256 \times 256 \times 160$ voxels, vượt quá dung lượng VRAM của GPU.

* **Kỹ thuật trích xuất Patch:** Cắt ngẫu nhiên các khối lập phương nhỏ (ví dụ $64 \times 64 \times 64$).

* **Biased Sampling:** Đảm bảo ít nhất $50\%$ số patch lấy mẫu có chứa ít nhất một phần voxel của khối u, $50\%$ còn lại lấy ngẫu nhiên khắp toàn bộ thể tích.

# HỌC PHẦN 2: AI FOR MEDICAL PROGNOSIS (TIÊN LƯỢNG Y KHOA)

Dự đoán rủi ro (Risk Score), thời gian sống còn, hoặc khả năng tái phát bệnh trong tương lai dựa trên dữ liệu lâm sàng dạng bảng (Tabular Clinical Data).

## 1. Dữ liệu bảng lâm sàng & Rừng quyết định

### 1.1. Bản chất dữ liệu khuyết thiếu (Missing Data)

* **MCAR (Missing Completely at Random):** Dữ liệu bị khuyết ngẫu nhiên do nguyên nhân kỹ thuật không liên quan đến bệnh nhân (ví dụ: máy in kết quả hết giấy, mẫu máu bị đổ). Loại bỏ mẫu hoặc điền trung vị không gây lệch phân phối.

* **MAR (Missing at Random):** Xác suất khuyết phụ thuộc vào một biến quan sát được khác.

  * *Ví dụ:* Bệnh nhân dưới 30 tuổi ít khi được bác sĩ chỉ định đo đường huyết hơn người trên 65 tuổi. Có thể dùng mô hình hồi quy dựa trên Tuổi để điền khuyết.

* **MNAR (Missing Not at Random):** Xác suất khuyết phụ thuộc trực tiếp vào chính giá trị của biến đó.

  * *Ví dụ:* Những bệnh nhân đang suy hô hấp nặng hoặc huyết áp tụt sâu thường không thể trả lời bảng câu hỏi khảo sát mức độ đau. Nếu điền giá trị bình thường, mô hình sẽ đánh giá thấp mức độ trầm trọng của bệnh.

### 1.2. Đánh giá khả năng xếp hạng rủi ro: C-Index (Concordance Index)

* Các mô hình tiên lượng không chỉ phân loại nhị phân mà xếp hạng mức độ rủi ro giữa các cá thể.

* **Cặp hợp lệ (Permissible Pair):** Hai bệnh nhân $(A, B)$ tạo thành một cặp hợp lệ khi ta biết chắc chắn người nào gặp biến cố trước.

* **Định nghĩa C-Index:** 

  $$
  \text{C-Index} = \frac{\text{Số cặp đồng thuận (Concordant)} + 0.5 \times \text{Số cặp hòa (Ties)}}{\text{Tổng số cặp hợp lệ (Permissible)}}
  $$

  * Một cặp là đồng thuận (Concordant) nếu bệnh nhân gặp biến cố sớm hơn được mô hình gán điểm rủi ro (Risk Score) cao hơn.

  * $C = 0.5$: Đoán ngẫu nhiên (không có giá trị lâm sàng).

  * $C = 1.0$: Khả năng xếp hạng rủi ro hoàn hảo.

## 2. Phân tích sống còn (Survival Analysis)

### 2.1. Hiện tượng Right-Censoring (Dữ liệu bị cắt ngọn)

Trong các nghiên cứu dài hạn, nhiều bệnh nhân không quan sát được biến cố (tử vong hoặc tái phát) tại thời điểm kết thúc nghiên cứu vì:

1. Nghiên cứu kết thúc mà bệnh nhân vẫn sống khỏe mạnh.

2. Bệnh nhân chuyển nơi ở, mất liên lạc (Lost to follow-up).

3. Bệnh nhân rút khỏi nghiên cứu vì lý do cá nhân.

```
Bệnh nhân 1: |================> [Biến cố: Tử vong năm thứ 3]        (Event = 1)
Bệnh nhân 2: |==========================> [Rút lui năm thứ 4]        (Event = 0, Censored)
Bệnh nhân 3: |=========================================> [Hết hạn]   (Event = 0, Censored)
             0                2               4               6 (Năm)

```

> **Nguyên tắc phân tích:** Tuyệt đối không xóa bệnh nhân bị Censored ra khỏi tập dữ liệu (gây mất mát thông tin và thiên lệch sống sót - Survivorship Bias), cũng không được coi họ là người sống vô tận.

### 2.2. Đường cong sống còn phi tham số Kaplan-Meier

Hàm sống sót $S(t) = P(T > t)$ biểu thị xác suất một người sống sót qua thời điểm $t$. Công thức tích lũy tích sống sót tại từng mốc thời gian xảy ra biến cố:

$$
S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)
$$

* $t_i$: Mốc thời gian xảy ra ít nhất 1 biến cố tử vong.

* $d_i$: Số ca tử vong ghi nhận tại thời điểm $t_i$.

* $n_i$: Số cá thể có nguy cơ (At-risk) ngay trước thời điểm $t_i$.

* **Bài toán thực hành:** Theo dõi 5 bệnh nhân ung thư phổi, thời gian ghi nhận (đơn vị: tháng): 

  $$
  t = [2, 3^+, 5, 7^+, 8]
  $$

   *(Dấu* $+$ *ký hiệu bệnh nhân bị Right-Censored)*

  1. **Tại** $t = 2$**:** $n_1 = 5$, $d_1 = 1$. 

     $$
     S(2) = 1 - \frac{1}{5} = 0.80
     $$

  2. **Tại** $t = 3^+$**:** Bệnh nhân bị mất dấu theo dõi (Censored). Số lượng người có nguy cơ giảm đi 1, nhưng tại thời điểm này không có ai tử vong ($d=0$), nên $S(3) = 0.80$.

  3. **Tại** $t = 5$**:** Số người còn nguy cơ $n_2 = 3$ (gồm bệnh nhân ở mốc 5, 7+, 8). Có $d_2 = 1$ người tử vong. 

     $$
     S(5) = S(2) \times \left(1 - \frac{1}{3}\right) = 0.80 \times 0.667 = 0.533
     $$

### 2.3. Mô hình rủi ro tỷ lệ Cox (Cox Proportional Hazards - Cox PH)

Hàm Hazard $\lambda(t)$ thể hiện nguy cơ tức thời xảy ra biến cố tại thời điểm $t$, với điều kiện bệnh nhân đã sống sót đến $t$:

$$
\lambda(t \vert{} X) = \lambda_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p)
$$

* $\lambda_0(t)$: Baseline Hazard (nguy cơ nền theo thời gian, không cần giả định dạng hàm toán học).

* $\exp(\beta^T X)$: Hazard Ratio (tỷ số rủi ro do các biến lâm sàng gây ra, bất biến theo thời gian).

* **Diễn giải hệ số lâm sàng:**

  * Giả sử biến $X_1$ là "Tăng huyết áp" ($1$: Có, $0$: Không), với hệ số hồi quy học được là $\beta_1 = 0.693$.

  * Tỷ số rủi ro: $\text{HR} = e^{0.693} \approx 2.0$.

  * Ý nghĩa: Bệnh nhân tăng huyết áp có nguy cơ tử vong tức thời cao gấp $2$ lần bệnh nhân bình thường tại bất kỳ mốc thời gian nào.

### 2.4. Mạng Deep Survival (DeepSurv)

* Mở rộng mô hình Cox PH cho các mối quan hệ phi tuyến tính phức tạp bằng mạng nơ-ron sâu: $h(X) = f_\theta(X)$.

* **Cox Partial Likelihood Loss (Hàm mất mát hợp lý từng phần):**

  $$
  L(\theta) = - \sum_{i: E_i = 1} \left( f_\theta(X_i) - \log \sum_{j \in R(T_i)} \exp(f_\theta(X_j)) \right)
  $$

  * $E_i = 1$: Chỉ tính tổng trên những bệnh nhân thực sự xảy ra biến cố.

  * $R(T_i)$: Tập hợp tất cả các bệnh nhân còn sống ở thời điểm $T_i$ (Risk set).

# HỌC PHẦN 3: AI FOR MEDICAL TREATMENT (ĐIỀU TRỊ & THỬ NGHIỆM LÂM SÀNG)

Ứng dụng suy luận nhân quả (Causal Inference) để tối ưu hóa quyết định điều trị cá nhân hóa và áp dụng NLP xử lý hồ sơ bệnh án điện tử (EHR).

## 1. Suy luận nhân quả & Phác đồ điều trị cá nhân hóa

### 1.1. Thử nghiệm lâm sàng ngẫu nhiên (RCT) vs. Dữ liệu quan sát (Observational Data)

* **RCT (Randomized Controlled Trial):** Bệnh nhân được phân bổ ngẫu nhiên vào nhóm Can thiệp (Treatment $W=1$) hoặc Đối chứng (Control $W=0$). Phân bổ ngẫu nhiên triệt tiêu sự mất cân bằng của các yếu tố nhiễu tiềm ẩn (Confounders).

* **Observational Data (Dữ liệu quan sát thực tế):** Bác sĩ chỉ định thuốc dựa trên tình trạng nặng nhẹ của bệnh nhân.

  * *Nhiễu do chỉ định (Confounding by Indication):* Bệnh nhân nặng thường được ưu tiên dùng thuốc kháng sinh thế hệ mới liều cao. Nếu so sánh trực tiếp, nhóm dùng thuốc mới có tỷ lệ tử vong cao hơn nhóm dùng thuốc cũ. Kết luận "thuốc mới gây tử vong" là sai, vì **mức độ nặng ban đầu của bệnh nhân** chính là biến gây nhiễu chi phối cả quyết định điều trị và tỷ lệ tử vong.

### 1.2. Khung kết quả tiềm năng (Neyman-Rubin Potential Outcomes)

Đối với mỗi cá thể $i$, luôn tồn tại 2 kết quả tiềm năng:

* $Y_i(1)$: Kết quả sức khỏe nếu được điều trị ($W=1$).

* $Y_i(0)$: Kết quả sức khỏe nếu không điều trị ($W=0$).

* **Vấn đề cơ bản của Suy luận nhân quả (Fundamental Problem of Causal Inference):** Ta không bao giờ quan sát được đồng thời cả $Y_i(1)$ và $Y_i(0)$ trên cùng một bệnh nhân tại cùng một thời điểm. Một kết quả là thực tế, kết quả còn lại là **Counterfactual** (ngược thực tế).

### 1.3. Ước lượng hiệu quả điều trị

* **ATE (Average Treatment Effect - Hiệu quả can thiệp trung bình):** 

  $$
  \text{ATE} = \mathbb{E}[Y(1) - Y(0)]
  $$

* **CATE (Conditional Average Treatment Effect - Hiệu quả điều trị có điều kiện / ITE):** 

  $$
  \text{CATE}(x) = \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
  $$

   *Ý nghĩa y học:* Một loại thuốc có thể có $\text{ATE} \approx 0$ (vô hiệu trên toàn bộ dân số nói chung), nhưng lại có $\text{CATE} > 0$ rất cao đối với phân nhóm bệnh nhân mang thụ thể đột biến gen đặc hiệu.

### 1.4. Các cấu trúc học điều trị cá nhân hóa

* **T-Learner (Two Learners):**

  * Chia dữ liệu làm 2 tập riêng:

    1. Huấn luyện mô hình $\mu_1(X)$ chỉ trên nhóm được điều trị ($W=1$).

    2. Huấn luyện mô hình $\mu_0(X)$ chỉ trên nhóm đối chứng ($W=0$).

  * Dự đoán CATE cho bệnh nhân mới: 

    $$
    \hat{\tau}(X) = \mu_1(X) - \mu_0(X)
    $$

* **S-Learner (Single Learner):**

  * Huấn luyện một mô hình máy học duy nhất nhận biến điều trị $W$ như một đặc trưng đầu vào cùng với các biến lâm sàng: $\mu(X, W)$.

  * Dự đoán CATE: 

    $$
    \hat{\tau}(X) = \mu(X, 1) - \mu(X, 0)
    $$

  * *Hạn chế của S-Learner:* Khi số chiều của biến đặc trưng $X$ rất lớn, các thuật toán (như Random Forest hoặc Gradient Boosting) có xu hướng bỏ qua hoặc xem nhẹ biến nhị phân $W$, khiến hiệu ứng điều trị ước tính bị suy giảm về $0$.

### 1.5. Propensity Score Matching & Weighting

* **Propensity Score (Điểm xu hướng** $e(X)$**):** Xác suất một bệnh nhân nhận điều trị dựa trên các đặc điểm nền: 

  $$
  e(X) = P(W = 1 \mid X)
  $$

* **Inverse Probability of Treatment Weighting (IPTW):** Dùng $e(X)$ làm trọng số để biến đổi tập dữ liệu quan sát thành một quần thể giả định cân bằng: 

  $$
  w_i = \frac{W_i}{e(X_i)} + \frac{1 - W_i}{1 - e(X_i)}
  $$

   Bệnh nhân nặng nhưng không được dùng thuốc, hoặc bệnh nhân nhẹ nhưng lại được dùng thuốc sẽ nhận trọng số lớn hơn để bù trừ độ lệch phân phối.

## 2. Xử lý ngôn ngữ tự nhiên (NLP) trên hồ sơ bệnh án (EHR)

### 2.1. Trích xuất thông tin y tế (Information Extraction)

* **Named Entity Recognition (NER - Nhận dạng thực thể y học):** Tự động phát hiện và gán nhãn các thực thể trong bệnh án:

  * Bệnh lý / Triệu chứng: *Pneumonia*, *Fever*.

  * Tên thuốc: *Metformin*, *Lisinopril*.

  * Liều dùng: *500mg daily*.

* **Relation Extraction (RE - Rút trích quan hệ):** Xác định liên kết ngữ nghĩa giữa các thực thể vừa tìm được:

  * Ví dụ: `[Metformin]` $\xrightarrow{\text{TREATS}}$ `[Type 2 Diabetes]`

  * Ví dụ: `[Lisinopril]` $\xrightarrow{\text{CAUSES\_ADVERSE\_EVENT}}$ `[Dry Cough]`

### 2.2. Nhận diện phủ định lâm sàng (Negation Detection) & Thuật toán NegEx

* **Vấn đề đặc thù:** Tên bệnh xuất hiện trong hồ sơ thường nằm trong câu phủ định kiểm tra loại trừ.

  * *Ví dụ:* *"Patient denies shortness of breath, no history of myocardial infarction."*

  * Nếu dùng tìm kiếm từ khóa thuần túy, bệnh nhân sẽ bị dán nhãn nhầm là có nhồi máu cơ tim.

* **Cấu trúc thuật toán NegEx:**

  * **Pre-negation triggers:** Cụm từ phủ định đứng trước: *"denies", "negative for", "rules out", "no signs of"*.

  * **Post-negation triggers:** Cụm từ phủ định đứng sau: *"was excluded", "unlikely"*.

  * **Scope of Negation:** Cửa sổ ảnh hưởng của từ khóa phủ định (thường từ 3 đến 6 từ tiếp theo hoặc cho đến khi gặp dấu ngắt câu / liên từ đảo ngược như *"but", "however"*).

### 2.3. Mô hình ngôn ngữ chuyên sâu: BioBERT & ClinicalBERT

* **Hạn chế của BERT thông thường:** Huấn luyện trên Wikipedia và BookCorpus, không nắm vững từ ngữ chuyên ngành và cách viết tắt lâm sàng.

* **Pre-training thích ứng miền (Domain Adaptation):**

  * **BioBERT:** Tiếp tục huấn luyện BERT trên cơ sở dữ liệu y sinh khổng lồ: bài báo nghiên cứu PubMed và PubMed Central.

  * **ClinicalBERT:** Huấn luyện trực tiếp trên toàn bộ văn bản ghi chú lâm sàng phi cấu trúc của ICU từ cơ sở dữ liệu chuẩn quốc tế **MIMIC-III**.

* Giúp giải quyết các từ viết tắt đa nghĩa tùy chuyên khoa:

  * *pt*: Patient (Bệnh nhân) hoặc Pint (Đơn vị đo).

  * *SOB*: Shortness of Breath (Khó thở).

  * *OD*: Once Daily (Mỗi ngày một lần) hoặc Overdose (Quá liều) hoặc Right Eye (Mắt phải).

# BẢNG TỔNG HỢP KIẾN THỨC TOÀN KHÓA

| **Chuyên đề** | **Thách thức cốt lõi** | **Kỹ thuật / Thuật toán chủ đạo** | **Thước đo đánh giá trọng tâm** | 
| **Diagnosis (Chẩn đoán)** | Extreme Imbalance, Patient ID Leakage, Voxel mô nhỏ | Weighted Loss, GroupSplit, 3D U-Net, Patch Sampling | AUROC, Sensitivity, Specificity, Dice Score | 
| **Prognosis (Tiên lượng)** | Right-Censoring, Missing Data (MAR/MNAR) | Kaplan-Meier, Cox PH, DeepSurv, Random Survival Forests | Concordance Index (C-Index), Brier Score | 
| **Treatment (Điều trị)** | Confounding Bias, Ngược thực tế (Counterfactual) | T-Learner, S-Learner, Propensity Score (IPTW) | CATE, Precision-Recall for NER | 
