# Recommendation_System

## Mô tả bài toán và ý tưởng
1. Mô tả bài toán:
  Với một tập các phim mà một người dùng đã xem và đánh giá, hệ thống cần phải xác định (dự đoán) những bộ phim nào (chưa được xem) mà người dùng đó thích xem.

2. Ý tưởng:
  Hai người xem những bộ phim giống nhau, đánh giá các phim đó giống nhau thì sẽ có xu hướng thích cùng các bộ phim khác trong tương lai; người này sẽ thích những bộ phim mà người kia đã đánh giá cao và ngược lại.

## Đôi nét về Dataset
  Bộ dữ liệu chúng ta sử dụng là bộ dữ liệu movielens dataset, đc upload tại https://grouplens.org/datasets/movielens/. Bộ dữ liệu này có rất nhiều option cho chúng ta lựa chọn nhưng do chúng ta chỉ đáp ứng cho nhu cầu education and development nên chỉ sử dụng bộ dữ liệu movieslens 100k để tối ưu thời gian tính toán và test mô hình. 
  
Trong bộ dữ liệu gồm các thành phần sau: 
* link.csv: Chứa movieId, idmbId và tmbId
* tags.csv: Chứa userId, movieId, tag, timestamp
* movies.csv: Chứa movieId, title, generes
* rating.csv: Chứa userId, movieId, rating, timestamp

Do đây là bộ dữ liệu dùng để education và development nên grouplens đã preprocessing hết cho chúng ta và do yêu cầu bài toán nên ta chỉ cần sử dụng 2 file là movies.csv và rating.csv 

## Mô hình sử dụng
Chúng ta sẽ sử dụng Matrix Factorization via Singular Value Decomposition cho hệ thống movies_recommendation này. Tôi không sử dụng user-based and item-based collaborative filtering vì 2 lý do chính sau:
* Nó k sử dụng tốt trên một data lớn, đặc biệt là trên hệ thống thời gian thực based trên mối tương quan về hành vi của các người dùng vì nó cần thực hiện rất nhiều các phép toán.
* Ma trận rating rất dễ bị overfit do các nhiễu về khẩu vị hay các mối quan tâm của người dùng.

Vì vậy chúng ta cần apply **Dimensionality Reduction** technique để reduction dimension

Singular Value Decomposition là việc phân tích một ma trận Am×n bất kỳ đều có thể phân tích thành dạng:
<img src="https://upanh1.com/images/Capture18d044fc35d383b1.png">
* R: là ma trận rating của ng dùng
* U: là user "features" matrix
* sigma: là ma trận đường chéo của Singular Value. 
* Vt: là movie "features" matrix.
Trong đó U và Vt là các ma trận trực giao và biểu diễn những điều khác nhau. U diễn tả bn user like mỗi movie và Vt diễn tả mối tương quan giữa các user vs movie. 

## Bài Toán

### Setup Rating Data
Ta sẽ load 2 file csv là movies và rating và loại bỏ 2 cột k cần thiết là timestamp và gener.
<img src="https://upanh1.com/images/Capturef7b2b0cc0c1d1f5c.png">

Sau đó, ta sẽ cần tạo một ma trận raring với hàng là movieID, cột là userId và các giá trị trong đó là điểm rating của ng dùng đó ứng với movie tương ứng.Có một phương thức giúp ta làm điều này là pivot 
<img src="https://upanh1.com/images/Capture6973073fd34b2309.png">
<img src="https://upanh1.com/images/Capture9f1c20c6fe0863d0.png">'

Cuối cùng chúng ta cần phải de-normalize data (normalize bằng cách tính mean điểm ng đó rate trên các bộ phim) và convert về numpy array để tính toán. 

<img src="https://upanh1.com/images/Capture09a52dec976db489.png">

Với ma trận đã được chuẩn hóa thì ta sẽ đi vào thuật toán SVD.

### Singular Value Decomposition
Scipy and Numpy both have functions to do the singular value decomposition. I'm going to use the Scipy function svds because it let's me choose how many latent factors I want to use to approximate the original ratings matrix (instead of having to truncate it after).
Cả Scipy và Numpy đều có function SVD nhưng mình sẽ sử dụng SVD của Scipy vì nó scipy cho phép mình chọn các factor bên trong nó (ở đây là cho phép mình chọn k). Ở các bài toán nhỏ ntn thì k thường trong khoảng 25-100 là tốt nhất để cho lượng thông tin mất đi ở mức chấp nhận đc nhưng quá trình tính toán nhanh hơn rất nhiều. 

Do ma trận sigma đc return về dưới dạng value chứ k phải ma trận đường chéo nên ta phải convert nó về ma trận đường chéo. 

<img src="https://upanh1.com/images/Capture8a2c4175b6712cc9.png">
