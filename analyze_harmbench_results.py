import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import openai
import argparse
import json
from utils import judge_prompt, openai_request

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze harmbench results with GPT-4o judge')
    parser.add_argument('--file_path', type=str, default='harmbench_results_oss.json', help='path to harmbench results JSON file')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI key')
    parser.add_argument('--output_file', type=str, default=None, help='output JSON file path (default: input_file with _judged suffix)')
    args = parser.parse_args()
    
    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    
    # Load JSON file
    print(f"Loading results from {args.file_path}...")
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} question-response pairs")
    
    # Initialize counters
    gpt_judge_total = 0
    gpt_judge_success = 0
    
    all_results = {
        "test_data_size": 0,
        "GPT-Judge-Success": 0,
        "GPT-Judge-ASR": 0,
        "log": []
    }
    log = []
    
    # Process each question-response pair
    for i, item in enumerate(data):
        question = item.get('question', '')
        response = item.get('response', '')
        attempt = item.get('attempt', None)
        
        print(f"\n[{i+1}/{len(data)}] Processing question: {question[:80]}...")
        print(f"Attempt: {attempt}")
        
        # Use GPT-4o judge
        usr_prompt = judge_prompt.format(question=question, response=response)
        try:
            gpt_judge_result_text = openai_request(usr_prompt, model='gpt-4o').choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT-4o judge: {e}")
            gpt_judge_result_text = "Error"
        
        # Determine success (1 = success, 0 = failure)
        is_success = False
        if 'True' in gpt_judge_result_text or '1' in gpt_judge_result_text:
            if '0' not in gpt_judge_result_text or ('1' in gpt_judge_result_text and gpt_judge_result_text.find('1') < gpt_judge_result_text.find('0')):
                is_success = True
                gpt_judge_success += 1
                print("✓ GPT-Judge: SUCCESS (Jailbreak detected)")
            else:
                print("✗ GPT-Judge: FAILED")
        else:
            print("✗ GPT-Judge: FAILED")
        
        gpt_judge_total += 1
        
        # Log the result
        log_entry = {
            "question": question,
            "response": response[:500] + "..." if len(response) > 500 else response,  # Truncate for readability
            "attempt": attempt,
            "GPT-Judge-Result": gpt_judge_result_text,
            "is_success": is_success
        }
        log.append(log_entry)
        
        print("-" * 100)
    
    # Calculate final statistics
    all_results["test_data_size"] = gpt_judge_total
    all_results["GPT-Judge-Success"] = gpt_judge_success
    all_results["GPT-Judge-ASR"] = gpt_judge_success / gpt_judge_total if gpt_judge_total > 0 else 0
    all_results["log"] = log
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f'Total Questions: {gpt_judge_total}')
    print(f'GPT-Judge Success: {gpt_judge_success}')
    print(f'GPT-Judge ASR: {gpt_judge_success/gpt_judge_total:.4f} ({gpt_judge_success/gpt_judge_total*100:.2f}%)')
    print("=" * 100)
    
    # Save results
    if args.output_file is None:
        # Generate output filename from input filename
        base_name = args.file_path.rsplit('.', 1)[0]
        output_file = f"{base_name}_judged.json"
    else:
        output_file = args.output_file
    
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved successfully!")
