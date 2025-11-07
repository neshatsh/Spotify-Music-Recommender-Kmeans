"""
Create User Preferences from Dataset
Samples diverse songs from our actual dataset
"""

import pandas as pd

df = pd.read_csv('genres_v2.csv')

print(f"Total songs: {len(df)}")
print("\nTop 10 genres:")
print(df['genre'].value_counts().head(10))

# Select diverse genres (avoiding Dark Trap & Underground Rap dominance)
selected_genres = ['Hiphop', 'trance', 'techno', 'RnB', 'Rap', 'Pop', 'Emo']

print("Creating preferences from these genres:")
print(', '.join(selected_genres))

preferences = []

for genre in selected_genres:
    genre_songs = df[df['genre'] == genre]
    
    if len(genre_songs) > 0:
        # Sample 3 random songs from this genre
        n = min(3, len(genre_songs))
        sampled = genre_songs.sample(n=n, random_state=42)
        preferences.append(sampled)
        print(f"{genre}: {n} songs")

# Combine
user_prefs = pd.concat(preferences, ignore_index=True)

user_prefs.to_csv('my_preferences.csv', index=False)

print(f"Created my_preferences.csv with {len(user_prefs)} songs")

print("\nGenre distribution:")
print(user_prefs['genre'].value_counts())

