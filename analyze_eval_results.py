#!/usr/bin/env python3
"""
Evaluation Results Analysis Script
분석할 CSV 파일의 경로를 입력하면 자동으로 분석 결과를 출력합니다.
"""

import pandas as pd
import argparse
import os
import sys

def analyze_eval_results(eval_csv_path, responses_csv_path=None):
    """
    평가 결과 CSV 파일을 분석합니다.
    
    Args:
        eval_csv_path: 평가 결과 CSV 파일 경로
        responses_csv_path: 응답 결과 CSV 파일 경로 (선택사항)
    """
    
    print(f"=== {eval_csv_path} 분석 시작 ===\n")
    
    # 파일 존재 확인
    if not os.path.exists(eval_csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {eval_csv_path}")
        return
    
    # 평가 결과 데이터 로드
    try:
        eval_df = pd.read_csv(eval_csv_path)
    except Exception as e:
        print(f"❌ CSV 파일 읽기 오류: {e}")
        return
    
    # 응답 결과 데이터 로드 (선택사항)
    responses_df = None
    if responses_csv_path and os.path.exists(responses_csv_path):
        try:
            responses_df = pd.read_csv(responses_csv_path)
        except Exception as e:
            print(f"⚠️ 응답 파일 읽기 오류: {e}")
    
    print('=== 기본 정보 ===')
    print(f'총 템플릿 수: {len(eval_df)}')
    if responses_df is not None:
        print(f'총 질문 수: {len(responses_df)}')
    print()
    
    print('=== 성공한 템플릿 분석 ===')
    successful_templates = eval_df[eval_df['success_q'] > 0]
    print(f'성공한 템플릿 수: {len(successful_templates)}')
    print(f'성공률: {len(successful_templates)/len(eval_df)*100:.1f}%')
    print()
    
    print('=== 쿼리 수 분석 ===')
    print('전체 템플릿의 쿼리 수 분포:')
    print(eval_df['total_num_query'].describe())
    print()
    
    print('성공한 템플릿의 쿼리 수 분포:')
    print(successful_templates['total_num_query'].describe())
    print()
    
    print('=== 질문당 평균 쿼리 수 계산 ===')
    total_queries = eval_df['total_num_query'].max()
    total_questions = 156  # 기본 질문 수 (test set 크기)
    if responses_df is not None:
        total_questions = len(responses_df)
        print(f'전체 질문 수: {total_questions} (실제 응답된 질문 수)')
    else:
        print(f'전체 질문 수: {total_questions} (기본 test set 크기)')
    
    avg_queries_per_question = total_queries / total_questions
    print(f'전체 쿼리 수: {total_queries}')
    print(f'질문당 평균 쿼리 수: {avg_queries_per_question:.2f}')
    print()
    
    # print('=== 템플릿별 쿼리 수 분석 ===')
    # template_queries = eval_df['total_num_query'].tolist()
    # template_queries.insert(0, 0)
    
    # queries_per_template = []
    # for i in range(len(template_queries)-1):
    #     queries_for_this_template = template_queries[i+1] - template_queries[i]
    #     queries_per_template.append(queries_for_this_template)
    
    # # 양수 쿼리 수만 분석
    # positive_queries = [q for q in queries_per_template if q > 0]
    # print(f'양수 쿼리 수: {len(positive_queries)}개')
    # print(f'음수 쿼리 수: {len(queries_per_template) - len(positive_queries)}개')
    # print()
    
    # print('양수 쿼리 수 분포:')
    # print(f'  평균: {sum(positive_queries)/len(positive_queries):.1f}')
    # print(f'  최대: {max(positive_queries)}')
    # print(f'  최소: {min(positive_queries)}')
    
    # # 50번 제한 관련 분석
    # near_50 = [q for q in positive_queries if q >= 45 and q <= 55]
    # over_50 = [q for q in positive_queries if q > 50]
    # print(f'  45-55 범위: {len(near_50)}개')
    # print(f'  50 초과: {len(over_50)}개')
    # print()
    
    # print('=== 성공한 질문 수 분석 ===')
    # print('성공한 질문 수 분포:')
    # print(eval_df['success_q'].describe())
    # print()
    
    # print('성공한 질문 수별 템플릿 개수:')
    # success_counts = eval_df['success_q'].value_counts().sort_index()
    # for success_num, count in success_counts.items():
    #     print(f'  {success_num}개 질문 성공: {count}개 템플릿')
    # print()
    
    # print('=== Mutator 분석 ===')
    # print('사용된 Mutator 분포:')
    # mutator_counts = eval_df['mutation'].value_counts()
    # for mutator, count in mutator_counts.items():
    #     print(f'  {mutator}: {count}개 템플릿')
    # print()
    
    # print('=== Generation 분석 ===')
    # print('세대별 분포:')
    # generation_counts = eval_df['generation'].value_counts().sort_index()
    # for gen, count in generation_counts.items():
    #     print(f'  Generation {gen}: {count}개 템플릿')
    # print()
    
    # print('=== 성공 패턴 분석 ===')
    # # 성공한 질문 수가 높은 템플릿들
    # high_success = eval_df[eval_df['success_q'] >= 2]
    # print(f'2개 이상 질문을 성공시킨 템플릿: {len(high_success)}개')
    # if len(high_success) > 0:
    #     print('상세:')
    #     for _, row in high_success.iterrows():
    #         print(f'  {row["success_q"]}개 성공, {row["total_num_query"]} 쿼리, {row["mutation"]}')
    # print()
    
    # print('=== Mutator별 성공률 ===')
    # for mutator in eval_df['mutation'].unique():
    #     mutator_data = eval_df[eval_df['mutation'] == mutator]
    #     avg_success = mutator_data['success_q'].mean()
    #     print(f'{mutator}: 평균 {avg_success:.2f}개 질문 성공 ({len(mutator_data)}개 템플릿)')
    # print()
    
    # print('=== 분석 완료 ===')

def main():
    parser = argparse.ArgumentParser(description='평가 결과 CSV 파일을 분석합니다.')
    parser.add_argument('eval_csv', help='평가 결과 CSV 파일 경로')
    parser.add_argument('--responses_csv', help='응답 결과 CSV 파일 경로 (선택사항)')
    
    args = parser.parse_args()
    
    analyze_eval_results(args.eval_csv, args.responses_csv)

if __name__ == "__main__":
    main()
