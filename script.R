#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


###########################################
# train algorithm
###########################################

#start with naive assumption (guess the mean)

mu <- mean(edx$rating)

#add regularized genre specific effect with concatinated genres 
#(regularization parameters determined previously)

b_g <- edx %>%
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu)*(n()/(n() + 10.7)))

#add regularized movie specific effect

b_m <- edx %>%
  left_join(b_g, by = "genres") %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu - b_g)*(n()/(n() + 4.4)))

#add regularized user specific effect

b_u <- edx %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_m, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_g - b_m)*(n()/(n() + 4.3)))

#cutpoints for time bins derived from whole movielens dataset (for use in time effects)
#creates 20 bins

cutpoints <- seq(min(edx$timestamp), max(edx$timestamp), length.out = 21)
cutpoints[1] <- cutpoints[1]-1

#add regularized movie-time effect to account for ratings changing 
#through time for individual movies

b_tm <- edx %>% 
  mutate(tbin = cut(timestamp, breaks = cutpoints, labels = FALSE)) %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(movieId) %>%
  group_by(tbin, add = TRUE) %>%
  summarize(b_tm = mean(rating - mu - b_g - b_m - b_u)*(n()/(n() + 51)))

#add regularized user-time effect to account for ratings changing 
#through time for individual users

b_tu <- edx %>% 
  mutate(tbin = cut(timestamp, breaks = cutpoints, labels = FALSE)) %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_tm, by = c("tbin", "movieId")) %>%
  group_by(userId) %>%
  group_by(tbin, add = TRUE) %>%
  summarize(b_tu = mean(rating - mu - b_g - b_m - b_u - b_tm)*(n()/(n() + 19)))


###############################
#make predictions on validation_set
###############################

#final predictions are limited to the range of possible outcomes
#(to correct for predictions that fall below 0.5 stars or above 5 stars)

test_pred <- 
  validation %>%
  mutate(tbin = cut(timestamp, breaks = cutpoints, labels = FALSE)) %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_tm, by = c("tbin", "movieId")) %>%
  left_join(b_tu, by = c("tbin", "userId")) %>%
  mutate(b_tm = if_else(is.na(b_tm) ,0, b_tm), 
         b_tu = if_else(is.na(b_tu), 0, b_tu),
         pred = mu + b_g + b_m + b_u + b_tm + b_tu,
         pred_round = case_when(pred <= .5 ~ .5001, 
                                pred > 5 ~ 5, 
                                TRUE ~ pred))

#make RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Calculate final RMSE
RMSE(test_pred$rating, test_pred$pred_round)
