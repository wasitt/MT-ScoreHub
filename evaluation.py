import nltk
import torch
import pandas as pd
import gdown
import subprocess
from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter
from nltk.translate.bleu_score import corpus_bleu as bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from sklearn.metrics import f1_score
from comet import download_model, load_from_checkpoint
import re

# Hugging Face 로그인
login("your_token")

# BLEURT 모델 로드
bleurt_model_name = "Elron/bleurt-base-128"
bleurt_tokenizer = AutoTokenizer.from_pretrained(bleurt_model_name)
bleurt_model = AutoModelForSequenceClassification.from_pretrained(bleurt_model_name)

# COMET 모델 로드
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

# nltk 호환성 다운로드
nltk.download('wordnet')

def evaluate_translations(input_file, sheet_index, output_file):
    """
    기계 번역 평가 점수를 계산하고 엑셀 파일로 저장하는 함수
    
    :param input_file: 입력 엑셀 파일 경로
    :param sheet_index: 사용할 시트 번호
    :param output_file: 출력 엑셀 파일 이름
    """
    # 데이터 로드
    data = pd.read_excel(input_file, sheet_name=sheet_index)
    
    # 데이터 추출
    original = data.iloc[:, 0]  # 원문 (중국어)
    translated = data.iloc[:, 1].tolist()  # 기계 번역 결과 (한국어)
    reference = data.iloc[:, 2].tolist()  # 참조 번역 (한국어)

    # 결과 데이터프레임 생성
    df_results = pd.DataFrame({
        "Original": original,
        "Translated": translated,
        "Reference": reference
    })

    # BLEU (NLTK) - Smoothing 적용
    smooth_func = SmoothingFunction().method1
    df_results["BLEU"] = [
        bleu([[ref.split()]], [hyp.split()], smoothing_function=smooth_func)
        for ref, hyp in zip(reference, translated)
    ]

    # SacreBLEU
    df_results["SacreBLEU"] = [
        corpus_bleu([hyp], [[ref]]).score
        for ref, hyp in zip(reference, translated)
    ]

    # chrF (SacreBLEU)
    df_results["ChrF"] = [
        corpus_chrf([hyp], [[ref]]).score
        for ref, hyp in zip(reference, translated)
    ]

    # TER (Translation Edit Rate)
    df_results["TER"] = [
        corpus_ter([hyp], [[ref]]).score
        for ref, hyp in zip(reference, translated)
    ]

    # METEOR 점수 계산
    df_results["Meteor"] = [
        meteor_score([ref.split()], hyp.split())
        for ref, hyp in zip(reference, translated)
    ]

    # BERTScore
    P, R, F1 = bert_score(translated, reference, lang="ko")
    df_results["BERTScore"] = F1.tolist()

    # BLEURT 점수 계산
    def compute_bleurt(hypothesis, reference):
        inputs = bleurt_tokenizer(reference, hypothesis, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            scores = bleurt_model(**inputs).logits.squeeze().tolist()
        return scores

    df_results["BLEURT"] = [
        compute_bleurt(hyp, ref)
        for ref, hyp in zip(reference, translated)
    ]

    # COMET 점수 계산
    comet_inputs = [
        {"src": src, "ref": ref, "mt": hyp}
        for src, ref, hyp in zip(original.tolist(), reference, translated)
    ]

    comet_scores = comet_model.predict(comet_inputs)
    df_results["Comet"] = comet_scores["scores"]

    # 결과 엑셀 저장
    df_results.to_excel(output_file, index=False)
    print(f"✅ 평가 완료! 결과가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    note = input("""1. 입력파일은 반드시 .xlsx 또는 .csv 형태여야 합니다.
2. 입력파일의 첫 번째 열은 원문, 두 번째 열은 한국어 번역문, 세 번째 열은 한국어 참조문이어야 합니다.

확인하셨다면 '네'를 입력해주세요: """)
    
    if note.strip().lower() == "네":
        file_url = input("Google Drive 공유 링크를 입력해주세요: ")
        sheet_num = int(input("엑셀 파일의 시트 개수를 입력해주세요: "))

        # 파일 ID 추출
        file_id_match = re.search(r"(?<=/d/)[\w-]+", file_url)
        if file_id_match:
            file_id = file_id_match.group(0)
        else:
            raise ValueError("올바른 Google Drive 파일 링크를 입력해주세요.")

        # 파일 다운로드
        file_path = "data.xlsx"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

        # 시트별 평가 실행
        for i in range(sheet_num):
            evaluate_translations(file_path, i, f"{i+1}번시트_결과.xlsx")

        print("✅ 모든 시트 평가가 완료되었습니다.")