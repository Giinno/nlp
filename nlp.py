import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import ssl
import threading
import time


class DynamicNLPApp:
    def __init__(self, master):
        self.master = master
        master.title("Dynamic NLP Tasks")
        master.geometry("800x600")

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.configure_styles()
        self.create_widgets()
        self.is_processing = False
        self.download_nltk_data()

    def configure_styles(self):
        self.style.configure('TLabel', font=('Helvetica', 12))
        self.style.configure('TButton', font=('Helvetica', 12))
        self.style.configure('TNotebook', background='#f0f0f0')
        self.style.configure('TNotebook.Tab', padding=[10, 5], font=('Helvetica', 10))
        self.style.map('TNotebook.Tab', background=[('selected', '#4a7abc')])
        self.style.configure('TFrame', background='#f0f0f0')

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input text area
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)

        self.input_label = ttk.Label(input_frame, text="Enter text:")
        self.input_label.pack(side=tk.LEFT, padx=(0, 10))

        self.input_text = scrolledtext.ScrolledText(input_frame, height=5, font=('Helvetica', 10))
        self.input_text.pack(fill=tk.X, expand=True)

        # Start button with animation
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.toggle_processing,
                                       style='Animate.TButton')
        self.start_button.pack(pady=10)

        # Create a notebook for different NLP tasks
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Tokenization tab
        self.tokenize_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tokenize_frame, text='Tokenization')
        self.tokenize_output = scrolledtext.ScrolledText(self.tokenize_frame, height=10, font=('Helvetica', 10))
        self.tokenize_output.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # NER tab
        self.ner_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ner_frame, text='Named Entity Recognition')
        self.ner_output = scrolledtext.ScrolledText(self.ner_frame, height=10, font=('Helvetica', 10))
        self.ner_output.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Sentiment Analysis tab
        self.sentiment_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sentiment_frame, text='Sentiment Analysis')
        self.sentiment_output = scrolledtext.ScrolledText(self.sentiment_frame, height=10, font=('Helvetica', 10))
        self.sentiment_output.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.animate_button()

    def animate_button(self):
        colors = ['#4a7abc', '#5a8acc', '#6a9adc', '#7aaaec', '#8abafc']

        def change_color(index=0):
            self.start_button.configure(style='Animate.TButton')
            self.style.configure('Animate.TButton', background=colors[index])
            self.master.after(100, change_color, (index + 1) % len(colors))

        change_color()

    def download_nltk_data(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        def download():
            resources = [
                'punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                'averaged_perceptron_tagger_eng', 'maxent_ne_chunker',
                'words', 'vader_lexicon'
            ]
            for resource in resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    self.master.after(0, lambda: messagebox.showerror("Download Error",
                                                                      f"Error downloading {resource}: {str(e)}"))
                    return
            self.master.after(0, lambda: messagebox.showinfo("Download Complete",
                                                             "NLTK resources downloaded successfully."))

        messagebox.showinfo("Downloading NLTK Data", "Downloading required NLTK data. This may take a moment.")
        threading.Thread(target=download, daemon=True).start()

        self.sid = SentimentIntensityAnalyzer()

    def toggle_processing(self):
        self.is_processing = not self.is_processing
        if self.is_processing:
            self.start_button.configure(text="Stop Processing")
            self.input_text.bind('<KeyRelease>', self.on_input_change)
            self.process_current_text()
        else:
            self.start_button.configure(text="Start Processing")
            self.input_text.unbind('<KeyRelease>')

    def process_current_text(self):
        text = self.input_text.get("1.0", tk.END).strip()
        self.update_tokenization(text)
        self.update_ner(text)
        self.update_sentiment(text)

    def on_input_change(self, event):
        if self.is_processing:
            self.process_current_text()

    def update_tokenization(self, text):
        try:
            tokens = word_tokenize(text)
            self.tokenize_output.delete("1.0", tk.END)
            self.tokenize_output.insert(tk.END, f"Tokens:\n{tokens}")
            self.highlight_text(self.tokenize_output, "Tokens:", "blue")
        except LookupError as e:
            self.tokenize_output.delete("1.0", tk.END)
            self.tokenize_output.insert(tk.END,
                                        f"Error: Required NLTK resource not found. Please restart the application to download missing resources.\n\nDetails: {str(e)}")
            self.highlight_text(self.tokenize_output, "Error:", "red")
        except Exception as e:
            self.tokenize_output.delete("1.0", tk.END)
            self.tokenize_output.insert(tk.END, f"Error in tokenization: {str(e)}")
            self.highlight_text(self.tokenize_output, "Error", "red")

    def update_ner(self, text):
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            self.ner_output.delete("1.0", tk.END)
            self.ner_output.insert(tk.END, f"Named Entities:\n{named_entities}")
            self.highlight_text(self.ner_output, "Named Entities:", "blue")
        except LookupError as e:
            self.ner_output.delete("1.0", tk.END)
            self.ner_output.insert(tk.END,
                                   f"Error: Required NLTK resource not found. Please restart the application to download missing resources.\n\nDetails: {str(e)}")
            self.highlight_text(self.ner_output, "Error:", "red")
        except Exception as e:
            self.ner_output.delete("1.0", tk.END)
            self.ner_output.insert(tk.END, f"Error in NER: {str(e)}")
            self.highlight_text(self.ner_output, "Error", "red")

    def update_sentiment(self, text):
        try:
            sentiment_scores = self.sid.polarity_scores(text)
            self.sentiment_output.delete("1.0", tk.END)
            self.sentiment_output.insert(tk.END, f"Sentiment Scores:\n{sentiment_scores}")
            self.highlight_text(self.sentiment_output, "Sentiment Scores:", "blue")

            compound_score = sentiment_scores['compound']
            if compound_score >= 0.05:
                sentiment = "Positive 😊"
                color = "green"
            elif compound_score <= -0.05:
                sentiment = "Negative 😞"
                color = "red"
            else:
                sentiment = "Neutral 😐"
                color = "gray"

            self.sentiment_output.insert(tk.END, f"\n\nOverall Sentiment: {sentiment}")
            self.highlight_text(self.sentiment_output, "Overall Sentiment:", "blue")
            self.highlight_text(self.sentiment_output, sentiment, color)
        except LookupError as e:
            self.sentiment_output.delete("1.0", tk.END)
            self.sentiment_output.insert(tk.END,
                                         f"Error: Required NLTK resource not found. Please restart the application to download missing resources.\n\nDetails: {str(e)}")
            self.highlight_text(self.sentiment_output, "Error:", "red")
        except Exception as e:
            self.sentiment_output.delete("1.0", tk.END)
            self.sentiment_output.insert(tk.END, f"Error in sentiment analysis: {str(e)}")
            self.highlight_text(self.sentiment_output, "Error", "red")

    def highlight_text(self, widget, text, color):
        start = "1.0"
        while True:
            start = widget.search(text, start, tk.END)
            if not start:
                break
            end = f"{start}+{len(text)}c"
            widget.tag_add(color, start, end)
            widget.tag_config(color, foreground=color)
            start = end


if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicNLPApp(root)
    root.mainloop()
