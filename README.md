# Machine_learing_final_report
## Cách lấy dữ liệu
Dữ liệu có thể tải trực tiếp từ github hoặc lấy qua link [https://archive.ics.uci.edu/dataset/963/ur3+cobotops](url)
## Hướng dẫn tổ chức thư mục
Các file nguồn đều được viết theo lập trình hướng đối tượng, cụ thể bao gồm:
* pca_Object.py, lda_Object.py là 2 đối tượng dùng để giảm chiều dữ liệu.
* dbscan_Object.py, kmeans_Object.py là 2 đối tượng dùng để phân cụm dữ liệu.
* naive_Bayes_Object.py, perceptron_Object.py và logistic_Regression_Object.py là các đối tượng dùng để phân loại và dự đoán máy móc đã cần bảo trì hay chưa. Trong đó naive_Bayes_Object là class cha bên trong gồm 3 class con là GaussianNB, MultinomialNB và BernoulliNB.
* confusionMatrixPlot.py dùng để vẽ ma trận nhầm lẫn.
* draw_Graph.py dùng để trực quan hóa dữ liệu.
* Ngoài ra 2 file thực thi chính là demo_Code_Use_Numpy.ipynb sử dụng các đối tượng ở trên để xử lý bài toán và demo_Code_Use_Library.ipynb có chức năng tương tự nhưng sử dụng các thư viện của Python.
Việc chuẩn hóa hay tiền xử lý dữ liệu được thực hiện trong 2 file thực thi.
Khuyến khích sắp xếp các file trong cùng một thư mục. Chú ý sửa lại đường dẫn cho phù hợp.
## Trân trọng!
