# Dự án dịch: "Deep Learning Interviews"

-[XEM TẠI ĐÂY](https://www.overleaf.com/read/scjxpdxbtggp#9c5f82)

## 📚 Giới thiệu
🎉 Cuốn sách này là kho tàng với hàng trăm bài toán Học Sâu kèm lời giải, giúp sinh viên, nhà nghiên cứu, và người chuẩn bị phỏng vấn AI tự tin hơn. Chúng mình muốn mang kiến thức này đến cộng đồng Việt Nam qua bản dịch dễ hiểu! 🚀

## 🎯 Mục tiêu

- **Dịch thuật**: Chuyển ngữ chính xác, lưu dưới dạng LaTeX (tương thích Overleaf)
- **Cộng tác**: Mời bạn cùng góp sức để bản dịch thêm hoàn hảo

Sách được thiết kế cho sinh viên Thạc sĩ/Tiến sĩ và người đi phỏng vấn, với các bài toán thực tế và câu hỏi sâu sắc.

## 📑 Cấu trúc sách
Ở mỗi chương tác giả sẽ chia làm hai phần lớn
- I. Các câu hỏi/bài tập: Tại đây, có thể chia làm các mục nhỏ hơn theo chủ đề.
- II. Lời giải/đáp án: Tương tự, sẽ được chia làm các mục nhỏ hơn.
  
Cuốn sách bao gồm các phần chính sau:

### Phần I: Rusty Nail
- Hướng dẫn sử dụng sách

### Phần II: Probabilistic Programming & Bayesian DL
## Chương 1. HỒI QUY LOGISTIC (LOGISTIC REGRESSION)

**Khái niệm cốt lõi:**
- **Hàm Sigmoid/Softmax:** Ánh xạ phi tuyến từ $\mathbb{R} \to [0,1]$ với $\sigma(z) = \frac{1}{1 + e^{-z}}$, softmax là dạng tổng quát $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$
- **Tỷ lệ cược (Odds ratio) & Log-odds:** $\text{odds} = \frac{p}{1-p}$, $\text{logit}(p) = \ln\frac{p}{1-p}$ - cơ sở của mô hình phân biệt (discriminative model)
- **Hàm mất mát (Binary Cross-entropy):** $\mathcal{L}(\hat{y}, y) = -[y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$
- **Tối ưu hóa:** Sử dụng thuật toán Gradient descent với cách cập nhật tham số $\theta := \theta - \eta\nabla_{\theta}\mathcal{L}(\theta)$
- **Mô hình tuyến tính tổng quát (GLM):** Bao gồm (i) phân phối xác suất (Bernoulli/Binomial), (ii) hàm tuyến tính đầu vào $\eta = X\beta$, (iii) hàm liên kết (logit)
- **Các chỉ số đánh giá:** Ma trận nhầm lẫn (confusion matrix), độ chính xác (accuracy), độ chính xác (precision), độ thu hồi (recall), điểm F1 (F1-score), ROC-AUC, log-loss

**Ứng dụng chính:**
- **Dịch tễ học & Thử nghiệm lâm sàng:** Dự đoán tiên lượng (prognosis), phân tích tỷ lệ nguy cơ (hazard ratio) và nguy cơ tương đối (relative risk)
- **Tài chính định lượng:** Chấm điểm tín dụng (credit scoring), đánh giá rủi ro, phát hiện gian lận
- **Xử lý ngôn ngữ tự nhiên (NLP):** Phân loại văn bản, phân tích cảm xúc (kết hợp với word embeddings)
- **Khoa học thần kinh tính toán:** Mô hình hóa neuron sinh học với ngưỡng kích hoạt (activation thresholds)
- **Mô hình cơ sở cho học tập tổng hợp:** Độ quan trọng đặc trưng (feature importance) và khả năng giải thích mô hình (model interpretability)

## Chương 2. LẬP TRÌNH XÁC SUẤT & HỌC SÂU BAYESIAN

**Khái niệm cốt lõi:**
- **Định lý Bayes:** $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ - cập nhật thông tin tiên nghiệm (prior beliefs) với dữ liệu đã được chứng thực (evidence)
- **Ước lượng hợp lý cực đại (MLE):
- **Ước lượng hậu nghiệm cực đại (MAP):
- **Phân phối liên hợp (Conjugate priors):** Cặp phân phối tiên nghiệm-hậu nghiệm cùng họ: Beta-Binomial, Dirichlet-Multinomial, Normal-Normal
- **Ma trận thông tin Fisher:** $\mathcal{I}(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\ln f(X;\theta)\right)^2\right]$ - đo lường độ cong của log-likelihood
- **Suy luận biến phân (Variational Inference):** Xấp xỉ phân phối hậu nghiệm $p(z|x)$ với phân phối $q_{\phi}(z|x)$ bằng cách tối thiểu KL divergence

**Ứng dụng chính:**
- **Định lượng độ không chắc chắn:** Dự đoán độ không chắc chắn trong chẩn đoán y tế và hệ thống tự hành
- **Mạng nơ-ron Bayesian (BNN):** Sử dụng phân phối trọng số thay vì ước lượng điểm để tăng tính ổn định (robustness)
- **Framework lập trình xác suất:** PyMC3, Stan, Pyro cho mô hình hóa sinh (generative modeling) và suy luận nhân quả (causal inference)
- **Tối ưu hóa Bayesian:** Điều chỉnh siêu tham số với các hàm tìm kiếm (acquisition functions) như Expected Improvement, UCB
- **Phương pháp Monte Carlo chuỗi Markov (MCMC):** Lấy mẫu hậu nghiệm trong không gian ẩn nhiều chiều

### Phần III: High School
## Chương 3. LÝ THUYẾT THÔNG TIN

**Khái niệm cốt lõi:**
- **Entropy Shannon:** $H(X) = -\sum_{x \in \mathcal{X}} p(x)\log_2 p(x)$ - đo lường nội dung thông tin/độ không chắc chắn
- **Độ đo phân kỳ Kullback-Leibler (KL Divergence):** $D_{KL}(P||Q) = \sum_{x \in \mathcal{X}}P(x)\log\frac{P(x)}{Q(x)}$ - độ đo bất đối xứng của sự khác biệt giữa 2 phân phối
- **Thông tin tương hỗ (MI):** $I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = D_{KL}(P_{X,Y}||P_X \otimes P_Y)$
- **Bất đẳng thức Jensen:** Với hàm $f$ lồi, $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ - nền tảng cho nhiều giới hạn trong tối ưu hóa
- **Entropy có điều kiện:** $H(X|Y) = -\sum_{x,y} p(x,y)\log p(x|y)$ - entropy còn lại sau khi quan sát Y

**Ứng dụng chính:**
- **Tối ưu hóa mạng nơ-ron:** Cross-entropy và KL-divergence làm hàm mất mát (loss functions)
- **Nút thắt thông tin (Information Bottleneck):** Cân bằng giữa nén và dự đoán trong học biểu diễn (representation learning)
- **Lý thuyết tốc độ-biến dạng, nhiễu thông tin (Rate-distortion):** Lý thuyết mã hóa và nén có mất mát với ứng dụng trong VAEs
- **Lựa chọn đặc trưng:** Tối đa hóa thông tin tương hỗ giữa đặc trưng và nhãn mục tiêu trong dữ liệu nhiều chiều
- **Điều chuẩn dựa trên lý thuyết thông tin:** Kiểm soát luồng thông tin trong GAN và thích ứng miền (domain adaptation)

## Chương 4. GIẢI TÍCH & VI PHÂN THUẬT TOÁN

**Khái niệm cốt lõi:**
- **Lan truyền ngược (Backpropagation):** Lập trình động cho tính toán gradient trên đồ thị tính toán với độ phức tạp thời gian $O(n)$
- **Quy tắc chuỗi cho hàm nhiều biến:** $\frac{\partial z}{\partial x_i} = \sum_{j} \frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}$ - nền tảng toán học của backprop
- **Phương pháp tối ưu bậc nhất:** $\theta_{t+1} = \theta_t - \eta_t \nabla_{\theta}\mathcal{L}(\theta_t)$ với các biến thể như momentum, Nesterov, AdaGrad, RMSProp, Adam
- **Đồ thị có hướng không chu trình (DAG):** Biểu diễn tính toán xuôi và dòng gradient ngược trong mạng nơ-ron
- **Vi phân tự động (Automatic Differentiation):** Biểu diễn số kép (dual-number), chế độ xuôi/ngược trong các framework như PyTorch, TensorFlow, JAX

**Ứng dụng chính:**
- **Huấn luyện mạng nơ-ron sâu:** Tính toán gradient hiệu quả trên mô hình quy mô lớn
- **Tối ưu hóa bậc hai:** Xấp xỉ ma trận Hessian và ma trận thông tin Fisher cho hội tụ nhanh hơn
- **Tìm kiếm kiến trúc nơ-ron (NAS):** Sử dụng phương pháp dựa trên gradient cho tối ưu hóa kiến trúc
- **Meta-learning & Học ít mẫu:** Đạo hàm bậc cao hơn cho thích ứng dựa trên gradient
- **Lớp ẩn và lập trình vi phân:** Vật lý vi phân (differentiable physics), lớp tối ưu hóa và bộ giải số học

### Phần IV: Bachelors
## Chương 5. TẬP HỢP MẠNG NƠ-RON (NEURAL NETWORK ENSEMBLES)

**Khái niệm cốt lõi:**
- **Tổng hợp Bootstrap (Bagging):** $f_{\text{ensemble}}(x) = \frac{1}{M}\sum_{m=1}^{M}f_m(x)$ với mỗi $f_m$ được huấn luyện trên mẫu bootstrap
- **Thuật toán tăng cường (Boosting):** Huấn luyện tuần tự với tập trung vào các mẫu phân loại sai; AdaBoost: $F_t(x) = F_{t-1}(x) + \alpha_t h_t(x)$
- **Tổng quát hóa xếp tầng (Stacked Generalization):** Kiến trúc đa cấp kết hợp các bộ học cơ sở qua meta-learner: $f_{\text{meta}}(f_1(x), f_2(x), ..., f_k(x))$
- **Tập hợp snapshot (Snapshot Ensemble):** Lịch trình tỷ lệ học tập chu kỳ với $\alpha(t) = \frac{\alpha_0}{2}(1 + \cos(\frac{\pi \text{ mod}(t-1, c)}{c}))$ - lưu mô hình tại cực tiểu cục bộ

**Ứng dụng chính:**
- **Tăng cường độ chính xác dự đoán:** Giảm phương sai thông qua đa dạng hóa mô hình trong các cuộc thi kaggle
- **Phát hiện phân phối ngoại lai (Out-of-distribution):** Sự bất đồng của ensemble làm chỉ số không chắc chắn cho phát hiện bất thường
- **Cây quyết định tăng cường gradient (GBDT):** XGBoost, LightGBM cho dữ liệu dạng bảng với hiệu suất cao
- **Tích hợp đa phương thức (Multi-modal fusion):** Tổng hợp mô hình từ các phương thức đa dạng (văn bản, hình ảnh, chuỗi thời gian) cho dự đoán mạnh mẽ
- **Dropout Monte Carlo:** Giải thích Bayesian của dropout như suy luận ensemble

## Chương 6. TRÍCH XUẤT ĐẶC TRƯNG CNN

**Khái niệm cốt lõi:**
- **Embedding CNN đã huấn luyện trước:** Trích xuất đặc trưng từ các lớp trung gian của mô hình được huấn luyện trên ImageNet như VGG, ResNet, EfficientNet
- **Mô hình học chuyển giao (Transfer Learning):** Trích xuất đặc trưng, tinh chỉnh (fine-tuning), đóng băng tiến trình (progressive freezing), và chuyển giao kiến thức (knowledge distillation)
- **Trực quan hóa kích hoạt lớp:** Độ nhạy cảm che khuất (occlusion sensitivity), ánh xạ kích hoạt lớp theo gradient (Grad-CAM)

**Ứng dụng chính:**
- **Các tác vụ thị giác máy tính:** Phát hiện đối tượng (R-CNN, YOLO), phân đoạn ngữ nghĩa (U-Net, DeepLab)
- **Phân tích hình ảnh y tế:** Diễn giải X-quang, phân loại mẫu bệnh lý với thích ứng miền (domain adaptation)
- **Tìm kiếm hình ảnh dựa trên nội dung (CBIR):** Tìm kiếm tương đồng trong cơ sở dữ liệu hình ảnh sử dụng embedding CNN
- **Học ít mẫu và Meta-learning:** Mạng prototype và mạng Siamese cho các tình huống dữ liệu hạn chế
- **Embedding đa phương thức:** Biểu diễn kết hợp giữa đặc trưng hình ảnh và phương thức văn bản/âm thanh

## Chương 7. KHÁI NIỆM HỌC SÂU

**Khái niệm cốt lõi:**
- **Chiến lược kiểm chứng chéo:** $k$-fold, phân tầng (stratified), leave-one-out, phân chia chuỗi thời gian với cân bằng phương sai-độ lệch phù hợp
- **Thuật toán học perceptron:** $w_{t+1} = w_t + \eta (y - \hat{y})x$ với đảm bảo hội tụ cho dữ liệu tuyến tính tách biệt
- **Phép tích chập (Convolution):** $(f * g)(t) = \int f(\tau)g(t-\tau)d\tau$ hoặc dạng rời rạc: $(f * g)[n] = \sum_{m} f[m]g[n-m]$
- **Hàm kích hoạt:** ReLU: $f(x) = \max(0,x)$, GELU: $x\Phi(x)$ với $\Phi$ là CDF của phân phối chuẩn, Swish: $f(x) = x\sigma(\beta x)$
- **Chỉ số hiệu suất:** $\text{Precision} = \frac{TP}{TP+FP}$, $\text{Recall} = \frac{TP}{TP+FN}$, $\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Kỹ thuật điều chuẩn (Regularization):** Suy giảm trọng số ($L_1$/$L_2$), Dropout ($p$), Chuẩn hóa batch (Batch Normalization): $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$

**Ứng dụng chính:**
- **Kiến trúc thị giác máy tính:** CNN (ResNet, EfficientNet), Vision Transformer (ViT), DETR cho phát hiện đối tượng
- **Mô hình NLP:** Kiến trúc Transformer (BERT, GPT), cơ chế chú ý (attention): $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- **Dự báo chuỗi thời gian:** Kiến trúc hồi quy (LSTM, GRU) với cơ chế cổng (gating) và cơ chế chú ý
- **Học tự giám sát (Self-supervised learning):** Phương pháp đối nghịch (SimCLR, CLIP), autoencoder có mặt nạ, bootstrap đặc trưng ẩn của bạn (BYOL)
- **Mô hình hóa sinh (Generative modeling):** VAE với giới hạn dưới bằng chứng (ELBO), GAN với huấn luyện đối kháng, mô hình khuếch tán (diffusion)

### Phần V: Practice Exam
- Bài kiểm tra mô phỏng phỏng vấn: Perceptrons, CNN Layers, Logistic Regression

### Phần VI: Volume Two (Kế hoạch)
- Thiết kế hệ thống AI, CNN nâng cao, NLP, GANs, RL



<!-- 
- docs/: Bản dịch LaTeX theo chương, sẵn cho Overleaf
- progress.md: Nhật ký tiến độ dịch
- CONTRIBUTING.md: Hướng dẫn đóng góp (sắp có)
- **Ước lượng hợp lý cực đại (MLE):** $ \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^{n}P(x_i|\theta)$ hoặc $\arg\max_{\theta} \sum_{i=1}^{n}\ln P(x_i|\theta)$
- **Ước lượng hậu nghiệm cực đại (MAP):** $\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(\theta|X) = \arg\max_{\theta} [P(X|\theta)P(\theta)]$
-->

<!--
## 🤝 Tham gia đóng góp

Muốn cùng dịch sách? Dễ thôi:

1. Fork repo này
2. Dịch hoặc sửa file LaTeX (Overleaf-friendly!)
3. Gửi Pull Request để tụi mình xem

Có ý tưởng? Mở Issue để trò chuyện nhé! 💬
-->


---

📝 *Bản dịch phi lợi nhuận, tôn trọng bản quyền tác giả Shlomo Kashani.*
