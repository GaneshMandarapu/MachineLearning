

#distance calculation function 
setwd("S:\\ANALYTICS\\R_Programming\\Clustering")

k_Data <- read.csv("Data.csv")

head(k_Data)

dim(k_Data[1,])


x = k_Data

#Random samples for centers 
#centers <- k_Data[sample(nrow(k_Data), 3),]


centers <- k_Data[18:19,]




# 
# euclid <- function(points1, points2) {
#   distanceMatrix <- matrix(NA, nrow=dim(points1)[1], ncol=dim(points2)[1])
#   for(i in 1:nrow(points2)) {
#     distanceMatrix[,i] <- sqrt(rowSums(t(t(points1)-points2[i,])^2))
#   }
#   distanceMatrix
# }
# 
# sqrt(rowSums(t(t(points1)-points2[1,])^2))



euclidnew <- function(x, centers) {
  distanceMatrix <- matrix(NA, nrow=nrow(x), ncol=nrow(centers))
  for (j in 1:nrow(centers) ){
    for(i in 1:nrow(x) ) {
      distanceMatrix[i,j] <- sqrt(rowSums((x[i,]-centers[j,])^2))
    }
  }
  distanceMatrix
}



euclidnew(x, centers)




K_means <- function(x, centers, distFun, nItter) {
  clusterHistory <- vector(nItter, mode="list")
  centerHistory <- vector(nItter, mode="list")
  
  for(i in 1:nItter) {
    distsToCenters <- distFun(x, centers)
    clusters <- apply(distsToCenters, 1, which.min)
    centers <- apply(x, 2, tapply, clusters, mean)
    
    # Saving history
    clusterHistory[[i]] <- clusters
    centerHistory[[i]] <- centers
  }
  
  list(clusters=clusterHistory, centers=centerHistory)
}



Results <- K_means(x, centers, euclidnew, 8)


Results

Results$centers[[8]]

Results$clusters[[8]]



Orig_Results <- kmeans(x, centers, 10, algorithm="Forgy")

Orig_Results$centers

Orig_Results



par(mfrow=c(2,2))
for(i in 5:10) {
  plot(points1, col=Results$clusters[[i]], main=paste("iteration:", i), xlab="x", ylab="y")
  points(Results$centers[[i]], cex=3, pch=19, col=1:nrow(Results$centers[[i]]))
}








#Testing Purpose
distsToCenters <- euclidnew(points1, points2)

clusters <- apply(distsToCenters, 1, which.min)

centers <- apply(points1, 2, tapply, clusters, mean)




