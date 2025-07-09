import argparse
import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader
# import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset, load_from_disk
from accelerate import PartialState, Accelerator
from accelerate.utils import gather, gather_object, tqdm

torch.set_float32_matmul_precision('high')

def get_reward_scores(rm, rm_tokenizer, prompts, responses1, responses2, device):
    all_conv1 = []
    all_conv2 = []
    for prompt, resp1, resp2 in zip(prompts, responses1, responses2):
        conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp1}]
        conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp2}]
        conv1_formatted = rm_tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = rm_tokenizer.apply_chat_template(conv2, tokenize=False)
        all_conv1.append(conv1_formatted)
        all_conv2.append(conv2_formatted)

    # Format and tokenize the conversations
    conv1_tokenized = rm_tokenizer.batch_encode_plus(all_conv1, return_tensors="pt", padding=True).to(device)
    conv2_tokenized = rm_tokenizer.batch_encode_plus(all_conv2, return_tensors="pt", padding=True).to(device)

    # Get the reward scores
    with torch.no_grad():
        score1 = rm(**conv1_tokenized).logits[:,0].to('cpu', torch.float32)
        score2 = rm(**conv2_tokenized).logits[:,0].to('cpu', torch.float32)
    return score1, score2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Gold Labeling')
    parser.add_argument('--dataset_fp', type=str, default='models/EleutherAI/pythia-160m-deduped/sft_model_2/dataset_full')
    parser.add_argument('--local_batch_size', type=int, default=8)

    args = parser.parse_args()
    # Load model and tokenizer
    dataset = load_from_disk(args.dataset_fp)
    dataset = dataset.with_format("torch", columns=['query', 'response1', 'response2', 'reference_response', 'gen_id'])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size)

    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm.eval()
    if len(dataset) > 10000:
        rm = torch.compile(rm)
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    accelerator = Accelerator(even_batches=False)
    rm, dataloader = accelerator.prepare(rm, dataloader)
    
    lm_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b-deduped')
    lm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
    # response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
    # response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."


    total_data_dicts = []
    for batch in tqdm(dataloader, desc='Computing gold reward scores', total=len(dataloader)):
        # with distributed_state.split_between_processes(batch) as split_batch:
        queries = [b.replace('[PAD]','') for b in batch['query']]
        score1, score2 = get_reward_scores(rm, rm_tokenizer, queries, batch['response1'], batch['response2'], accelerator.device)
        if accelerator.is_main_process:
            print('input===', queries[0])
            print('response1===', batch['response1'][0])
            print('response2===', batch['response2'][0])
            print('reference_response===', batch['reference_response'][0])
            print(f'Scores: {score1[0]:.2f}, {score2[0]:.2f}')
        total_data_dicts.append(
            {
                'score_1': score1.cpu().numpy(), 
                'score_2': score2.cpu().numpy(),
                'queries': queries,
                'gen_ids': batch['gen_id'].cpu().numpy()
            })

    accelerator.wait_for_everyone()
    total_data_dicts = gather_object(total_data_dicts)

    if accelerator.is_main_process:
        # Sort the gathered results by gen_id

        data_dict = {
            k: np.concatenate([d[k] for d in total_data_dicts], axis=0)
            for k in total_data_dicts[0].keys()
        }

        gen_ids = np.array(data_dict['gen_ids'])[:len(dataset)]
        index = np.argsort(gen_ids)
        reward_scores1 = np.array(data_dict['score_1'])[index]
        reward_scores2 = np.array(data_dict['score_2'])[index]
        queries_gathered_sorted = [data_dict['queries'][i] for i in index]
        
        winner = np.argmax([reward_scores1, reward_scores2], axis=0)
        print('Winner:', np.sum(winner == 0), np.sum(winner == 1))

        matches = [q_new == q_ref.replace('[PAD]','') for q_new, q_ref in zip(queries_gathered_sorted, dataset['query'])]
        assert all(matches), f"gathered queries not matching dataset, mismatched indices: {np.where(~np.array(matches))[0]}"

        query = dataset['query']
        query_token = dataset['query_token']

        chosen = np.where(winner == 0, dataset['response1'], dataset['response2'])
        chosen_token = np.where(winner[:, None] == 0, dataset['response1_token'], dataset['response2_token'])
        rejected = np.where(winner == 0, dataset['response2'], dataset['response1'])
        rejected_token = np.where(winner[:, None] == 0, dataset['response2_token'], dataset['response1_token'])
        chosen_goldrew = np.where(winner == 0, reward_scores1, reward_scores2)
        rejected_goldrew = np.where(winner == 0, reward_scores2, reward_scores1)
        query_chosen = [q.strip() + c for q, c in zip(query, chosen)]
        query_rejected = [q.strip() + r for q, r in zip(query, rejected)]
        query_chosen_token = np.concatenate([np.array(query_token), np.array(chosen_token)], axis=1)
        query_rejected_token = np.concatenate([np.array(query_token), np.array(rejected_token)], axis=1)
        
        use_numpy_save = False
        if use_numpy_save:
            out_dict = np.load(args.dataset_fp, allow_pickle=True).item()
            # check queries not shuffled:
            matches = [q_new == q_ref for q_new, q_ref in zip(query, out_dict['query'])]
            assert all(matches), f"queries not matched, mismatched indices: {np.where(~np.array(matches))[0]}"
            
            out_dict['winner'] = winner
            out_dict['chosen'] = chosen
            out_dict['chosen_token'] = chosen_token
            out_dict['rejected'] = rejected
            out_dict['rejected_token'] = rejected_token
            out_dict['chosen_goldrew'] = chosen_goldrew
            out_dict['rejected_goldrew'] = rejected_goldrew
            out_dict['query_chosen'] = query_chosen
            out_dict['query_chosen_token'] = query_chosen_token
            out_dict['query_rejected'] = query_rejected
            out_dict['query_rejected_token'] = query_rejected_token

            out_fp = args.dataset_fp + '_goldadded'
            out_dataset = Dataset.from_dict(out_dict)
            out_dataset.save_to_disk(out_fp, num_proc=1)
            print(f"Gold labeled dataset saved to {out_fp}")
        else:
            matches = [q_new == q_ref for q_new, q_ref in zip(query, dataset['query'])]
            assert all(matches), f"queries not matched, mismatched indices: {np.where(~np.array(matches))[0]}"
            dataset = dataset.add_column('winner', winner)
            dataset = dataset.add_column('chosen', chosen)
            dataset = dataset.add_column('chosen_token', list(chosen_token))
            dataset = dataset.add_column('rejected', rejected)
            dataset = dataset.add_column('rejected_token', list(rejected_token))
            dataset = dataset.add_column('chosen_goldrew', chosen_goldrew)
            dataset = dataset.add_column('rejected_goldrew', rejected_goldrew)
            dataset = dataset.add_column('query_chosen', query_chosen)
            dataset = dataset.add_column('query_rejected', query_rejected)
            dataset = dataset.add_column('query_chosen_token', list(query_chosen_token))
            dataset = dataset.add_column('query_rejected_token', list(query_rejected_token))
            dataset.save_to_disk(args.dataset_fp + '_goldadded', num_proc=1)
            print(f"Gold labeled dataset saved to {args.dataset_fp + '_goldadded'}")

    # accelerator.wait_for_everyone()

    # Output:
    # Score for response 1: 12.625
    # Score for response 2: -15.25

