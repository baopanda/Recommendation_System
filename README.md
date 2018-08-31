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
Chúng ta sẽ sử dụng Matrix Factorization và Singular Value Decomposition cho hệ thống movies_recommendation này.
