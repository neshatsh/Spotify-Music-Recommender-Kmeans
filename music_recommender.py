"""
Music Recommendation System
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class MusicRecommender:
    
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo'
        ]
        self.data = None
        self.scaled_features = None
        self.user_preference_genres = []
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        print(f" Total genres: {self.data['genre'].nunique()}")
        return self.data
    
    def balance_dataset(self, max_per_genre=300, exclude_genres=None):
        """Balance dataset to prevent genre dominance"""
        print("\n Balancing dataset...")
        
        if exclude_genres:
            print(f"   Excluding genres: {', '.join(exclude_genres)}")
            self.data = self.data[~self.data['genre'].isin(exclude_genres)]
        
        initial_size = len(self.data)
        balanced_dfs = []
        
        genre_counts = self.data['genre'].value_counts()
        print(f"   Before balancing - Top genres:")
        for genre, count in genre_counts.head(5).items():
            print(f"     {genre}: {count} songs")
        
        for genre in self.data['genre'].unique():
            genre_data = self.data[self.data['genre'] == genre]
            sample_size = min(len(genre_data), max_per_genre)
            balanced_dfs.append(genre_data.sample(n=sample_size, random_state=42))
        
        self.data = pd.concat(balanced_dfs, ignore_index=True)
        print(f"\n  Balanced: {initial_size} → {len(self.data)} songs")
        print(f"    Genres represented: {self.data['genre'].nunique()}")
        
        genre_counts_after = self.data['genre'].value_counts()
        print(f"   After balancing - Top genres:")
        for genre, count in genre_counts_after.head(5).items():
            print(f"     {genre}: {count} songs")
    
    def preprocess_data(self):
        available_features = [col for col in self.feature_columns if col in self.data.columns]
        self.feature_columns = available_features
        print(f"   Using {len(self.feature_columns)} features")
        
        # Remove duplicates
        initial_size = len(self.data)
        if 'song_name' in self.data.columns:
            self.data = self.data.drop_duplicates(subset=['song_name'], keep='first')
        print(f"   Removed {initial_size - len(self.data)} duplicates")
        
        # Handle missing values
        self.data = self.data.dropna(subset=self.feature_columns)
        
        # Normalize tempo and loudness
        if 'tempo' in self.data.columns:
            self.data['tempo'] = (self.data['tempo'] - self.data['tempo'].min()) / \
                                 (self.data['tempo'].max() - self.data['tempo'].min())
        
        if 'loudness' in self.data.columns:
            self.data['loudness'] = (self.data['loudness'] - self.data['loudness'].min()) / \
                                    (self.data['loudness'].max() - self.data['loudness'].min())
        
        # Standardize features
        self.scaled_features = self.scaler.fit_transform(self.data[self.feature_columns])
        
        print(f"    Final dataset: {len(self.data)} songs")
    
    def train_model(self):
        print(f"\n Training K-Means with {self.n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.data['cluster'] = self.kmeans.fit_predict(self.scaled_features)
        
        print("    Training complete!")
        print("\n   Cluster distribution:")
        for i in range(self.n_clusters):
            count = (self.data['cluster'] == i).sum()
            print(f"     Cluster {i}: {count} songs")
        
        self._analyze_clusters()
    
    def _analyze_clusters(self):
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            print(f"\n Cluster {cluster_id} ({len(cluster_data)} songs):")
            
            # Show top genres in this cluster
            if 'genre' in cluster_data.columns:
                top_genres = cluster_data['genre'].value_counts().head(3)
                print(f"   Top genres: {', '.join([f'{g} ({c})' for g, c in top_genres.items()])}")
            
            # Show audio characteristics
            feature_means = cluster_data[self.feature_columns].mean()
            print(f"   Energy: {feature_means.get('energy', 0):.2f}, "
                  f"Dance: {feature_means.get('danceability', 0):.2f}, "
                  f"Valence: {feature_means.get('valence', 0):.2f}")
    
    def generate_recommendations(self, user_prefs_path, output_dir='output'):
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS")
        print("="*60)
        
        # Load user preferences
        user_prefs = pd.read_csv(user_prefs_path)
        print(f"\n Loaded {len(user_prefs)} preference tracks")
        
        # Store user preference genres for diversity enforcement
        self.user_preference_genres = user_prefs['genre'].unique().tolist()
        print(f"   User prefers: {', '.join(self.user_preference_genres)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Predict clusters for preferences
        user_features = user_prefs[self.feature_columns].fillna(0)
        user_scaled = self.scaler.transform(user_features)
        user_clusters = self.kmeans.predict(user_scaled)
        
        print("\n   User preferences by cluster:")
        unique, counts = np.unique(user_clusters, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"     Cluster {cluster_id}: {count} songs")
        
        # Generate recommendations per cluster
        all_recommendations = []
        
        for cluster_id in range(self.n_clusters):
            cluster_songs = self.data[self.data['cluster'] == cluster_id].copy()
            
            # Rank by similarity with genre diversity
            ranked_songs = self._rank_with_diversity(cluster_songs, user_prefs, user_clusters, cluster_id)
            
            # Select top 5
            top_songs = ranked_songs.head(5)
            
            # Save playlist
            playlist_file = os.path.join(output_dir, f'playlist_cluster_{cluster_id}.csv')
            top_songs.to_csv(playlist_file, index=False)
            all_recommendations.append(top_songs)
            
            print(f"   Cluster {cluster_id}: Saved {len(top_songs)} songs")
        
        # Combined playlist with strong diversity enforcement
        combined = self._create_diverse_combined_playlist(all_recommendations)
        
        combined_file = os.path.join(output_dir, 'playlist_combined.csv')
        combined.to_csv(combined_file, index=False)
        
        print(f"\n Saved combined playlist with {len(combined)} songs")
        
        # Show final genre distribution
        print("\n Final Genre Distribution:")
        final_genres = combined['genre'].value_counts()
        for genre, count in final_genres.items():
            pct = (count / len(combined)) * 100
            print(f"   {genre}: {count} songs ({pct:.1f}%)")
        
        print("="*60)
    
    def _rank_with_diversity(self, cluster_songs, user_prefs, user_clusters, target_cluster):
        """Rank songs with genre diversity enforcement and preference matching"""
        
        # Calculate similarity scores
        similarities = []
        
        for idx, song in cluster_songs.iterrows():
            song_features = song[self.feature_columns].values
            
            # Find minimum distance to any preference song
            min_distance = float('inf')
            genre_match = False
            
            for _, pref_song in user_prefs.iterrows():
                pref_features = pref_song[self.feature_columns].values
                distance = np.sqrt(np.sum((song_features - pref_features) ** 2))
                min_distance = min(min_distance, distance)
                
                # Check genre match
                if song['genre'] == pref_song['genre']:
                    genre_match = True
            
            # Weight by cluster match
            if target_cluster in user_clusters:
                min_distance *= 0.7  # Boost if cluster matches
            
            # Boost for genre match with preferences
            if genre_match:
                min_distance *= 0.5  # Strong boost for matching genre
            
            similarity = 1 / (1 + min_distance)
            similarities.append(similarity)
        
        cluster_songs['similarity_score'] = similarities
        cluster_songs = cluster_songs.sort_values('similarity_score', ascending=False)
        
        # Enforce genre diversity: max 2 per genre in cluster
        if 'genre' in cluster_songs.columns:
            diverse_songs = []
            genre_counts = {}
            
            for idx, song in cluster_songs.iterrows():
                genre = song['genre']
                if genre_counts.get(genre, 0) < 2:
                    diverse_songs.append(song)
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
                if len(diverse_songs) >= 10:
                    break
            
            if diverse_songs:
                cluster_songs = pd.DataFrame(diverse_songs)
        
        return cluster_songs
    
    def _create_diverse_combined_playlist(self, all_recommendations):
        """Create combined playlist with strong genre diversity enforcement"""
        
        combined = pd.concat(all_recommendations, ignore_index=True)
        combined = combined.sort_values('similarity_score', ascending=False)
        
        diverse_songs = []
        genre_counts = {}
        
        # Maximum 4 songs per genre (for 25 total, ensures 6+ genres minimum)
        max_per_genre = 4
        
        # Priority 1: Include at least one song from each user preference genre
        for pref_genre in self.user_preference_genres:
            genre_songs = combined[combined['genre'] == pref_genre]
            if len(genre_songs) > 0 and genre_counts.get(pref_genre, 0) < max_per_genre:
                # Add best match from this preferred genre
                best_song = genre_songs.iloc[0]
                diverse_songs.append(best_song)
                genre_counts[pref_genre] = genre_counts.get(pref_genre, 0) + 1
                # Remove from pool
                combined = combined[combined.index != best_song.name]
        
        # Priority 2: Fill remaining slots with diversity constraint
        for _, song in combined.iterrows():
            genre = song['genre']
            if genre_counts.get(genre, 0) < max_per_genre:
                diverse_songs.append(song)
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            if len(diverse_songs) >= 25:
                break
        
        result = pd.DataFrame(diverse_songs)
        return result.sort_values('similarity_score', ascending=False)


def main():
    dataset_file = sys.argv[1]
    prefs_file = sys.argv[2]
    
    if not os.path.isfile(dataset_file):
        print(f"Error: {dataset_file} not found")
        sys.exit()
    
    if not os.path.isfile(prefs_file):
        print(f"Error: {prefs_file} not found")
        sys.exit()
    
    print("\n" + "="*60)
    print("MUSIC RECOMMENDATION SYSTEM")
    print("="*60)
    
    recommender = MusicRecommender(n_clusters=8)
    recommender.load_data(dataset_file)
    
    # Balance dataset to fix genre imbalance
    # Option 1: Exclude over-represented genres (uncomment if needed)
    # recommender.balance_dataset(max_per_genre=300, exclude_genres=['Dark Trap', 'Underground Rap'])
    
    # Option 2: Balance all genres (recommended)
    recommender.balance_dataset(max_per_genre=300)
    
    recommender.preprocess_data()
    recommender.train_model()
    recommender.generate_recommendations(prefs_file)
    
    print("\n Done! Check 'output/' folder for playlists\n")


if __name__ == "__main__":
    main()
