# Sentiment Analysis Jokowi dengan IndoBERT
# Menganalisis sentimen YouTube comments, dan berita

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Data collection libraries
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import feedparser

# NLP dan Machine Learning
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Utils
from datetime import datetime, timedelta
import json
import time
from collections import Counter

class JokowiSentimentAnalyzer:
    def __init__(self):
        """
        Inisialisasi analyzer dengan IndoBERT model
        """
        print("Memuat IndoBERT model...")
        # Menggunakan IndoBERT model untuk sentiment analysis
        self.model_name = "indolem/indobert-base-uncased"
        
        # Load pre-trained IndoBERT sentiment model (alternatif jika ada)
        try:
            # Coba gunakan model sentiment yang sudah fine-tuned
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
                tokenizer="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
            )
        except:
            # Fallback ke model umum
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
        
        print("Model berhasil dimuat!")
        
        # Inisialisasi data containers
        self.youtube_data = []
        self.news_data = []
        
    def clean_text(self, text):
        """
        Membersihkan teks dari karakter yang tidak diinginkan
        """
        if not isinstance(text, str):
            return ""
            
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Hapus mention dan hashtag (tapi simpan isinya)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Hapus karakter khusus tapi pertahankan emoticon
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
        
        # Hapus whitespace berlebih
        text = ' '.join(text.split())
        
        return text.strip()
    
    def collect_youtube_data(self, api_key, video_ids=None, count=2000):
        """
        Mengumpulkan komentar YouTube dengan error handling yang lebih baik
        """
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            if not video_ids:
                print("Mencari video tentang Jokowi...")
                # Search videos about Jokowi
                try:
                    search_response = youtube.search().list(
                        q='Jokowi',
                        part='id,snippet',
                        type='video',
                        maxResults=10,
                        order='relevance',
                        regionCode='ID',  # Indonesia region
                        relevanceLanguage='id'  # Indonesian language
                    ).execute()
                    
                    video_ids = []
                    for item in search_response['items']:
                        if 'videoId' in item['id']:
                            video_ids.append(item['id']['videoId'])
                            print(f"Found video: {item['snippet']['title'][:50]}... ID: {item['id']['videoId']}")
                    
                    if not video_ids:
                        print("Tidak ada video ditemukan, menggunakan sample data...")
                        self.generate_sample_youtube_data()
                        return
                        
                except Exception as e:
                    print(f"Error dalam pencarian video: {e}")
            
            print(f"Mengumpulkan komentar dari {len(video_ids)} video...")
            comments_per_video = max(1, count // len(video_ids))
            
            for i, video_id in enumerate(video_ids):
                print(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
                
                try:
                    # Cek apakah video exists dan comments enabled
                    video_response = youtube.videos().list(
                        part='statistics,snippet',
                        id=video_id
                    ).execute()
                    
                    if not video_response['items']:
                        print(f"Video {video_id} tidak ditemukan, skip...")
                        continue
                    
                    video_info = video_response['items'][0]
                    print(f"Video title: {video_info['snippet']['title'][:50]}...")
                    
                    # Cek apakah komentar diaktifkan
                    try:
                        comments_response = youtube.commentThreads().list(
                            part='snippet',
                            videoId=video_id,
                            maxResults=min(comments_per_video, 100),  # Max 100 per request
                            order='relevance',
                            textFormat='plainText'
                        ).execute()
                        
                        comments_found = 0
                        for comment_item in comments_response['items']:
                            try:
                                comment_snippet = comment_item['snippet']['topLevelComment']['snippet']
                                comment_text = comment_snippet['textDisplay']
                                
                                # Filter komentar yang relevan dengan Jokowi
                                if any(keyword in comment_text.lower() for keyword in ['jokowi', 'joko widodo', 'presiden']):
                                    self.youtube_data.append({
                                        'platform': 'YouTube',
                                        'text': self.clean_text(comment_text),
                                        'date': comment_snippet['publishedAt'],
                                        'author': comment_snippet.get('authorDisplayName', 'Unknown'),
                                        'likes': comment_snippet.get('likeCount', 0),
                                        'video_id': video_id,
                                        'video_title': video_info['snippet']['title']
                                    })
                                    comments_found += 1
                                    
                            except KeyError as e:
                                print(f"Error parsing comment structure: {e}")
                                continue
                            except Exception as e:
                                print(f"Error processing individual comment: {e}")
                                continue
                        
                        print(f"Collected {comments_found} relevant comments from video {video_id}")
                        
                    except Exception as e:
                        error_reason = str(e)
                        if "commentsDisabled" in error_reason:
                            print(f"Comments disabled for video {video_id}")
                        elif "quotaExceeded" in error_reason:
                            print("API quota exceeded, stopping collection")
                            break
                        else:
                            print(f"Error getting comments for video {video_id}: {e}")
                        continue
                        
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    continue
            
            total_comments = len(self.youtube_data)
            print(f"Berhasil mengumpulkan {total_comments} komentar YouTube")
            
            if total_comments == 0:
                print("Tidak ada komentar yang berhasil dikumpulkan, menggunakan sample data...")
                self.generate_sample_youtube_data()
                
        except Exception as e:
            print(f"Error umum dalam mengumpulkan data YouTube: {e}")
            print("Menggunakan sample data untuk demo...")
    
    def collect_news_data(self, sources=None, days_back=7):
        """
        Mengumpulkan data berita dari berbagai sumber
        """
        if not sources:
            sources = [
                'https://www.detik.com/tag/jokowi',
                'https://news.okezone.com/tag/jokowi',
                'https://www.kompas.com/tag/jokowi',
                'https://www.tempo.co/tag/jokowi'
            ]
        
        print("Mengumpulkan data berita...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for source in sources:
                try:
                    response = requests.get(source, headers=headers, timeout=100)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract article titles and snippets (customize based on website structure)
                    articles = soup.find_all(['h2', 'h3', 'h4'], limit=200)
                    
                    for article in articles:
                        title = article.get_text().strip()
                        if 'jokowi' in title.lower() and len(title) > 50:
                            self.news_data.append({
                                'platform': 'News',
                                'text': self.clean_text(title),
                                'date': datetime.now(),
                                'source': source,
                                'type': 'headline'
                            })
                            
                except Exception as e:
                    print(f"Error scraping {source}: {e}")
                    continue
                    
            print(f"Berhasil mengumpulkan {len(self.news_data)} berita")
            
        except Exception as e:
            print(f"Error mengumpulkan data berita: {e}")
    
    # def generate_sample_data(self):
    #     """
    #     Generate sample data untuk demo
    #     """
    #     print("Generating sample data untuk demo...")
    #     self.generate_sample_youtube_data()
    #     self.generate_sample_news_data()
    
    def analyze_sentiment(self, text):
        """
        Menganalisis sentimen menggunakan IndoBERT
        """
        try:
            result = self.sentiment_pipeline(text)
            
            # Standardize output format
            if isinstance(result, list):
                result = result[0]
            
            label = result['label'].upper()
            score = result['score']
            
            # Map different label formats to standard format
            if label in ['POSITIVE', 'POS', 'LABEL_2']:
                sentiment = 'POSITIVE'
            elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
                
            return {
                'sentiment': sentiment,
                'confidence': score,
                'raw_result': result
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.5,
                'raw_result': None
            }
    
    def process_all_data(self):
        """
        Memproses semua data dan melakukan analisis sentimen
        """
        print("Memproses dan menganalisis sentimen semua data...")
        
        all_data = []
        
        # Process YouTube data
        for item in self.youtube_data:
            if item['text']:
                sentiment_result = self.analyze_sentiment(item['text'])
                item.update(sentiment_result)
                all_data.append(item)
        
        # Process News data
        for item in self.news_data:
            if item['text']:
                sentiment_result = self.analyze_sentiment(item['text'])
                item.update(sentiment_result)
                all_data.append(item)
        
        self.df = pd.DataFrame(all_data)
        print(f"Total data yang diproses: {len(self.df)}")
        
        return self.df
    
    def generate_report(self):
        """
        Membuat laporan analisis sentimen
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("Tidak ada data untuk dianalisis. Jalankan process_all_data() terlebih dahulu.")
            return
        
        print("\n" + "="*60)
        print("LAPORAN ANALISIS SENTIMEN JOKOWI")
        print("="*60)
        
        # Overall statistics
        total_data = len(self.df)
        sentiment_counts = self.df['sentiment'].value_counts()
        
        print(f"\nTotal Data Dianalisis: {total_data}")
        print("\nDistribusi Sentimen:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count/total_data)*100
            print(f"- {sentiment}: {count} ({percentage:.1f}%)")
        
        # Platform breakdown
        print("\nDistribusi per Platform:")
        platform_sentiment = pd.crosstab(self.df['platform'], self.df['sentiment'])
        print(platform_sentiment)
        
        # Average confidence scores
        print(f"\nRata-rata Confidence Score: {self.df['confidence'].mean():.3f}")
        
        # Top positive and negative texts
        positive_texts = self.df[self.df['sentiment'] == 'POSITIVE'].nlargest(3, 'confidence')
        negative_texts = self.df[self.df['sentiment'] == 'NEGATIVE'].nlargest(3, 'confidence')
        
        print("\nTop 3 Sentimen Positif:")
        for idx, row in positive_texts.iterrows():
            print(f"- {row['text'][:100]}... (Confidence: {row['confidence']:.3f})")
        
        print("\nTop 3 Sentimen Negatif:")
        for idx, row in negative_texts.iterrows():
            print(f"- {row['text'][:100]}... (Confidence: {row['confidence']:.3f})")
    
    def create_visualizations(self):
        """
        Membuat visualisasi data
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("Tidak ada data untuk divisualisasikan.")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisis Sentimen Jokowi - Dashboard', fontsize=16, fontweight='bold')
        
        # Define color mapping for sentiments
        sentiment_colors = {
            'POSITIVE': '#2ecc71',  # Hijau untuk positif
            'NEGATIVE': '#e74c3c',  # Merah untuk negatif
            'NEUTRAL': '#95a5a6'    # Abu-abu untuk netral
        }
        
        # 1. Overall sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        # Map colors based on sentiment labels
        colors = [sentiment_colors.get(sentiment, '#95a5a6') for sentiment in sentiment_counts.index]
        
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Distribusi Sentimen Keseluruhan')
        
        # 2. Sentiment by platform
        platform_sentiment = pd.crosstab(self.df['platform'], self.df['sentiment'])
        # Create color list for bar chart based on column order
        bar_colors = [sentiment_colors.get(col, '#95a5a6') for col in platform_sentiment.columns]
        
        platform_sentiment.plot(kind='bar', ax=axes[0,1], color=bar_colors)
        axes[0,1].set_title('Sentimen per Platform')
        axes[0,1].set_xlabel('Platform')
        axes[0,1].set_ylabel('Jumlah')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(title='Sentimen')
        
        # Add percentage labels on bars
        total_per_platform = platform_sentiment.sum(axis=1)
        for i, platform in enumerate(platform_sentiment.index):
            cumulative = 0
            for j, sentiment in enumerate(platform_sentiment.columns):
                value = platform_sentiment.loc[platform, sentiment]
                if value > 0:
                    percentage = (value / total_per_platform[platform]) * 100
                    # Position label in the center of each bar segment
                    axes[0,1].text(i, cumulative + value/2, f'{percentage:.1f}%', 
                                  ha='center', va='center', fontweight='bold', fontsize=9)
                cumulative += value
        
        # 3. Confidence score distribution
        axes[1,0].hist(self.df['confidence'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1,0].axvline(self.df['confidence'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.df["confidence"].mean():.3f}')
        axes[1,0].set_title('Distribusi Confidence Score')
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 4. Sentiment timeline (if date available)
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
                daily_sentiment = self.df.groupby([self.df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                
                if len(daily_sentiment) > 1:
                    # Create color mapping for line plot
                    line_colors = [sentiment_colors.get(col, '#95a5a6') for col in daily_sentiment.columns]
                    daily_sentiment.plot(kind='line', ax=axes[1,1], marker='o', color=line_colors)
                    axes[1,1].set_title('Tren Sentimen dari Waktu ke Waktu')
                    axes[1,1].set_xlabel('Tanggal')
                    axes[1,1].set_ylabel('Jumlah')
                    axes[1,1].tick_params(axis='x', rotation=45)
                    axes[1,1].legend(title='Sentimen')
                else:
                    axes[1,1].text(0.5, 0.5, 'Data tidak cukup\nuntuk tren timeline', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
                    axes[1,1].set_title('Tren Sentimen (Data Terbatas)')
            except:
                axes[1,1].text(0.5, 0.5, 'Error dalam\nmemproses timeline', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Tren Sentimen (Error)')
        
        plt.tight_layout()
        plt.show()
        
        # Word cloud
        self.create_wordcloud()
    
    def create_wordcloud(self):
        """
        Membuat word cloud dari teks
        """
        if not hasattr(self, 'df') or self.df.empty:
            return
        
        # Combine all text
        all_text = ' '.join(self.df['text'].astype(str))
        
        # Create word cloud
        plt.figure(figsize=(12, 8))
        
        # Positive sentiment word cloud
        positive_text = ' '.join(self.df[self.df['sentiment'] == 'POSITIVE']['text'].astype(str))
        if positive_text.strip():
            plt.subplot(1, 2, 1)
            wordcloud_pos = WordCloud(width=400, height=400, background_color='white',
                                     colormap='Greens').generate(positive_text)
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.title('Word Cloud - Sentimen Positif', fontweight='bold')
            plt.axis('off')
        
        # Negative sentiment word cloud
        negative_text = ' '.join(self.df[self.df['sentiment'] == 'NEGATIVE']['text'].astype(str))
        if negative_text.strip():
            plt.subplot(1, 2, 2)
            wordcloud_neg = WordCloud(width=400, height=400, background_color='white',
                                     colormap='Reds').generate(negative_text)
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.title('Word Cloud - Sentimen Negatif', fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename='jokowi_sentiment_analysis.csv'):
        """
        Menyimpan hasil analisis ke file CSV
        """
        if hasattr(self, 'df') and not self.df.empty:
            self.df.to_csv(filename, index=False)
            print(f"Hasil analisis disimpan ke: {filename}")
        else:
            print("Tidak ada data untuk disimpan.")

# Contoh penggunaan
def main():
    """
    Fungsi utama untuk menjalankan analisis sentimen
    """
    print("Memulai Analisis Sentimen Jokowi dengan IndoBERT")
    print("="*50)
    
    # Inisialisasi analyzer
    analyzer = JokowiSentimentAnalyzer()

    
    # Untuk penggunaan dengan API real
    analyzer.collect_youtube_data(api_key="AIzaSyAnKKKIz0l1wa4ZEtcEtPdmJMMtMxYw1Pw")
    analyzer.collect_news_data()
    
    # Proses semua data
    df_results = analyzer.process_all_data()
    
    # Generate laporan
    analyzer.generate_report()
    
    # Buat visualisasi
    analyzer.create_visualizations()
    
    # Simpan hasil
    analyzer.save_results()
    
    print("\nAnalisis selesai!")
    
    return analyzer

if __name__ == "__main__":
    # Jalankan analisis
    sentiment_analyzer = main()
    
    # Akses data hasil analisis
    if hasattr(sentiment_analyzer, 'df'):
        print(f"\nData tersedia dalam variabel 'sentiment_analyzer.df' dengan {len(sentiment_analyzer.df)} baris")
        print("\nContoh data:")
        print(sentiment_analyzer.df[['platform', 'text', 'sentiment', 'confidence']].head())
