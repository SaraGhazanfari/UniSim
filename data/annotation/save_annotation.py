from data import BAPPSDataset, MagicBrush, HQEdit, Polaris
from data import KADID10kPairs, NightDataset, HPDv2Pairs, PieApp2AFCDataset, PiPal

from os.path import join
from torch.utils.data import ConcatDataset
import random, json
from data.annotation.constants import instruction_templates, answer_templates
import random, json
from os.path import join
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
import cv2
from PIL import Image

    
    
class SaveDataset:
    
    def __init__(self, data_dir, image_size=224, image_processor=None, text_processor=None, split='train', shuffle=False):
        hpd_v2 = HPDv2Pairs(data_dir=data_dir, split=split, instruct=None)
        
        datasets = [
            ConcatDataset([
            BAPPSDataset(root_dir=data_dir, split=split, load_size=image_size, image_processor=image_processor, ratio=0.25), 
            NightDataset(root_dir=data_dir, split=split, load_size=image_size, image_processor=image_processor),
            PieApp2AFCDataset(data_dir=data_dir, split=split, load_size=image_size, image_processor=image_processor)
            ]),
            ConcatDataset([
                MagicBrush(data_dir=data_dir, split=split),
                HQEdit(data_dir=data_dir, split=split, task_type='text-images-FC'), 
                Subset(hpd_v2, random.choices(range(len(hpd_v2)), k=400_000))  
            ]), 
            ConcatDataset([    #text-2AFC
                HQEdit(data_dir=data_dir, split=split, task_type='text_2afc'),
                Polaris(data_dir=data_dir, split=split),
            ]), 
            ConcatDataset([
                KADID10kPairs(root_dir=data_dir, sampling_mode='sev-noref', split=split),
                PiPal(data_dir=data_dir, split=split, image_processor=image_processor)
            ])
        ]
        self.dataloader_types = ['2afc', 'text_images_afc', 'text_2afc', 'iqa'] 
        self.datasets = self._process_dataset(datasets)
        self._save_annotation()
                
    def _save_annotation(self):
        name = 'iqa_unisim_instruct_annotation.json'
        
        with open(name, 'w') as file:
            json.dump(self.datasets, file)
            
        print(f'{name} saved successfully!')
        
    def _prepare_2afc_data(self, sample, instruction):
        ref, img_1, img_2, label, _ = sample
        
        ref_idx = random.randint(0,2)
        instruction = instruction[f'Image{ref_idx+1}']
        instruction = instruction[random.randint(0, len(instruction)-1)]
        
        if ref_idx == 0:
            images = [ref, img_1, img_2]
            options = ['2', '3']
        elif ref_idx == 1:
            images = [img_1, ref, img_2]
            options = ['1', '3']
        else:
            images = [img_1, img_2, ref]
            options = ['1', '2']
        return images, instruction, options
    
    def _prepare_answer(self, dataset_name, label, instruction, options=['1', '2']):
        label = options[int(label)]
        answers_list = answer_templates[dataset_name]
        
        if 'Options' in instruction and '(A)' in instruction:
            answer = 'B' if label in instruction.split('\n')[-1] else 'A'
        else:
            
            answer = answers_list[random.randint(0, len(answers_list)-1)].format(label=label)
        return answer
        
    def _process_dataset(self, datasets):
        total_len = len(datasets[0]) + len(datasets[1]) + len(datasets[2]) + len(datasets[3])
        print(len(datasets[0]), len(datasets[1]), len(datasets[2]), len(datasets[3]), total_len)
        processed_dataset = list()
        import time
        startt = time.time()
        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = self.dataloader_types[dataset_idx]
            
            for sample_idx, sample in enumerate(dataset):
                print(
                    f'dataset_name: {dataset_name}, dataset_idx: {dataset_idx}, sample_idx: {sample_idx}, time: {round(time.time()-startt, 4)}',
                    flush=True)
                instruction = instruction_templates[dataset_name] 
                options=['1', '2']
                if dataset_name == '2afc':
                    ref, img_1, img_2, label, _ = sample
                    images, instruction, options = self._prepare_2afc_data(sample, instruction)
                                        
                elif dataset_name == 'text_images_afc':
                    img_1, img_2, label, ref = sample
                    images = [img_1, img_2]
                    instruction = instruction[random.randint(0, len(instruction)-1)]
                    instruction = instruction.format(caption=ref)
                elif dataset_name =='text_2afc':
                    ref, cap_1, cap_2, label = sample
                    instruction = instruction[random.randint(0, len(instruction)-1)]
                    images = [ref]
                    instruction = instruction.format(caption1=cap_1, caption2=cap_2)
                    
                elif dataset_name == 'iqa':
                    img_1, img_2, label, _ = sample
                    instruction = instruction[random.randint(0, len(instruction)-1)]
                    images = [img_1, img_2]
                    
                else:
                    raise Exception('wrong dataset name')
            
                answer = self._prepare_answer(dataset_name, label, instruction, options=options)
                processed_dataset.append({'datasource': f'{dataset_name}',
                                           'id': f'{dataset_name}_{sample_idx}',
                                           'image': images,
                                           'conversations': [{'from': 'human','value': f'{instruction}'},
                                                             {'from': 'gpt','value': f'{answer}'}],
                                           'raw_label': int(label)})
                

       
        # print(len(processed_dataset))
        # random.shuffle(processed_dataset)
        print(len(processed_dataset))
        return processed_dataset
                
if __name__ == "__main__":
    SaveDataset(data_dir='/vast/sg7457/uni_data')
           
            


    

