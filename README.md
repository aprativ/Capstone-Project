### Capstone-Project
# CineMatch: A Content-Based Recommendation System for Movies and TV Shows!

Research Question: Can a content-based recommendation system utilizing features such as actors, directions, plot, number of seasons, age certification, production country, and genre, along with the popularity and score data from TMDB and IMDb, accurately predict a user’s rating of a movie or a TV Show. 

Data: Hulu, Prime, and Netflix movies/shows scrapped from the streaming guide called "Just Watch". Obtained from Kaggle :
  - Prime: https://www.kaggle.com/datasets/victorsoeiro/amazon-prime-tv-shows-and-movies
  + Netlfix: https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies
  * Hulu: https://www.kaggle.com/datasets/victorsoeiro/hulu-tv-shows-and-movies
  
  
Method of Analysis: 
  -	Feature Engineering: After the process of cleaning the data and fixing any discrepancies, the plan is to transform and preprocess the features in the dataset. For the text feature extraction, I plan on using Bag-of-words approach. Next, for the genre, age, production country, and director/actor variables, I will create a dummy variable for each of the different types. 
  +	Similarity measures: Calculate the similarity between the features of the movies/shows and recommend the ones with the highest score,using cosine similarity and Euclidean distance. 
  +	Clustering: Group shows/movies by similar features, using k-means clustering
  *	Matrix Factorization: Non-negative matrix factorization (NMF) to factorize matrix of movie or TV show features and their overall ratings.

Software: Python on Jupyter Notebook

References: 
  -	Choi, S.-M., Han, Y.-S.: A content recommendation system based on category correlations. In: The Fifth International Multi-Conference on Computing in the Global Information Technology, pp. 1257–1260 (2010)
  -	D. Das, H. T. Chidananda and L. Sahoo, "Personalized Movie Recommendation System Using Twitter Data" in Progress in Computing Analytics and Networking, Singapore:Springer, vol. 710, 2018.
  -	SK. Ko et al., "A Smart Movie Recommendation System", Lecture Notes in Computer Science, vol. 6771, 2011.
  -	N. Mishra, S. Chaturvedi, V. Mishra, R. Srivastava and P. Bargah, "Solving Sparsity Problem in Rating-Based Movie Recommendation System" in Computational Intelligence in Data Mining, Singapore:Springer, vol. 556, 2017.

