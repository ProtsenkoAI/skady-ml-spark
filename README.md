# Skady ML Spark
### It's fork of ML platform of Skady. It contains spark engine to fit Collaborative Filtering model on users interactions in realtime and asynchronously make partners recomendations for users.
### Main features are:
1. **Spark** workers are used to make recommendations and fit model, thus it's scallable
2. **Online fitting** when user itneracts with service, thus we make recomendatins from the very start
3. Using user features from [suvec project](https://github.com/ProtsenkoAI/skady-user-vectorizer) to gain more information without interactions
