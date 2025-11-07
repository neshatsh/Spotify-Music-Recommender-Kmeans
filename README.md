# Spotify Music Recommender

A K-Means clustering-based recommendation system that generates personalized Spotify playlists with balanced genre diversity.

## Features

- **Genre-Balanced Recommendations**: Prevents single genre dominance through dataset balancing
- **Spotify Audio Features**: Uses 9 Spotify audio features for similarity matching
- **Smart Similarity Scoring**: Euclidean distance with genre and cluster matching bonuses
- **Diversity Enforcement**: Ensures varied genres in recommendations
- **8-Cluster Architecture**: Optimal balance between specificity and diversity

## Quick Start
```bash
# Install dependencies
pip3 install -r requirements.txt

# Create user preferences from your Spotify dataset
python create_preferences.py

# Generate recommendations
python music_recommender.py genres_v2.csv my_preferences.csv
```

## Dataset

You can find genres_v2.csv dataset [here](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify) on kaggle.

Also the full list of genres included in the CSV are Trap, Techno, Techhouse, Trance, Psytrance, Dark Trap, DnB (drums and bass), Hardstyle, Underground Rap, Trap Metal, Emo, Rap, RnB, Pop and Hiphop.

You can find description of each feature in [here](https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features).


## Output

- `output/playlist_cluster_0.csv` through `playlist_cluster_7.csv` - 5 songs per cluster
- `output/playlist_combined.csv` - Combined diverse playlist

## How It Works

1. **Dataset Balancing**: Limits each genre to 300 songs to prevent dominance
2. **Feature Extraction**: Uses 9 Spotify audio features
3. **Clustering**: K-Means (k=8) groups similar tracks
4. **Recommendation**: 
   - Calculates similarity to user preferences using Euclidean distance


## Example Results

**Before Genre Balancing:**
- Dark Trap: 72% | RnB: 20% | Trap Metal: 8%

**After Genre Balancing:**
- Pop: 23.5% | RnB: 23.5% | Hiphop: 23.5% | Emo: 23.5% | Rap: 5.9%

## Technical Details

**Preprocessing:**
- Dataset balancing (max 300 songs per genre)
- Duplicate removal by song name
- Feature normalization (tempo, loudness → [0,1])
- StandardScaler standardization

**Similarity Measure:**
```python
distance = euclidean(song_features, preference_features)
if genre_match: distance *= 0.5  # 50% boost
if cluster_match: distance *= 0.7  # 30% boost
similarity = 1 / (1 + distance)
```

**Number of Clusters:** 8

