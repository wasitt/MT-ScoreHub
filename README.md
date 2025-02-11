## MT-ScoreHub 
MT-ScoreHub is a comprehensive machine translation evaluation tool that supports multiple translation quality assessment metrics, including BLEU, SacreBLEU, chrF, TER, METEOR, BERTScore, BLEURT, and COMET.

### Features
1. Supports multiple translation quality evaluation metrics.
2. Evaluates translations from an Excel (.xlsx) file.
3. Automatically downloads required models for BLEURT and COMET.
4. Outputs results to an Excel file for easy analysis.

### Installation and Usage
git clone https://github.com/your-username/MT-ScoreHub.git
pip install -r requirements.txt --quiet
python evaluation.py

### Output
The script generates an Excel file with the following evaluation metrics:
1. BLEU (NLTK-based score with smoothing)
2. SacreBLEU (Official BLEU score calculation)
3. chrF (Character n-gram F-score)
4. TER (Translation Edit Rate)
5. METEOR (Precision, Recall, and Score)
6. BERTScore (Semantic similarity based on embeddings)
7. BLEURT (Pre-trained model for MT evaluation)
8. COMET (State-of-the-art MT evaluation model)

Each sheet of the input file is evaluated separately, and results are saved as 1번시트_결과.xlsx, 2번시트_결과.xlsx, etc.
