from data import NightDataset, BAPPSDataset, BirdsToWords, MagicBrush, HQEdit, \
        HPDv2Pairs, Polaris, KADID10kPairs, KonIQ10kPairs, Foil, PiPal, PieApp2AFCDataset, PieAppIQADataset, SICEPairs
from os.path import join
from torch.utils.data import ConcatDataset, DataLoader, distributed, Subset
import random



class TrainingDataset:
    
    def __init__(self, args, image_size=224, image_processor=None, text_processor=None, split='train', shuffle=False):

        train_data_name_list = args.train_data_name.split(':')
        two_afc_data, text_images_afc_data, text_2afc_data, iqa_data = list(), list(), list(), list()
        
        if  'night' in train_data_name_list:
            two_afc_data.append(NightDataset(root_dir=args.train_data, split=split, load_size=image_size, image_processor=image_processor))
            
        if 'pie-app' in train_data_name_list:
            iqa_data.append(PieAppIQADataset(data_dir=args.train_data, split=split, load_size=image_size, image_processor=image_processor))
            two_afc_data.append(PieApp2AFCDataset(data_dir=args.train_data, split=split, load_size=image_size, image_processor=image_processor))
            
        if 'magic_brush' in train_data_name_list:
            text_images_afc_data.append(MagicBrush(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, split=split))
            
        if 'hq_edit' in train_data_name_list:
            text_images_afc_data.append(HQEdit(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, split=split, 
                                               task_type='text-images-FC'))
            text_2afc_data.append(HQEdit(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, 
                                        split=split, task_type='text_2afc'))
        if 'hpd' in train_data_name_list:
            text_images_afc_data.append(HPDv2Pairs(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, split=split,
                    instruct=None))
        if 'polaris':
            text_2afc_data.append(Polaris(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, split=split))
            
        if 'kadid' in train_data_name_list:
           iqa_data.append(KADID10kPairs(root_dir=args.train_data, transform_fn=image_processor, sampling_mode='sev-noref', split=split)) 
   
        if 'pipal' in train_data_name_list:
            iqa_data.append(PiPal(data_dir=args.train_data, split=split, image_processor=image_processor))
        
        self.datasets = [ConcatDataset(two_afc_data), ConcatDataset(text_images_afc_data), ConcatDataset(text_2afc_data), ConcatDataset(iqa_data)]
        
        self.num_samples = self._get_num_samples(num_samples_per_dataset=args.max_num_samples)
        self.dataloaders, self.num_batches, self.sampler_list = self._get_dataloader(batch_size=args.batch_size, 
                                                                                     num_workers=args.workers, shuffle=shuffle, 
                                                                                     world_size=args.world_size, rank=args.rank)
        self.dataloader_types = ['2AFC',
                                 'Text-Images-AFC',
                                 'Text-2AFC',
                                 'IQA'
                                 ] 
        self.running_avg_loss = [0] * len(self.dataloader_types)
    
    def _get_dataloader(self, batch_size, num_workers, shuffle, world_size, rank):
        
        dataloaders, sampler_list = list(), list()
        num_batches = 0
        for idx, dataset in enumerate(self.datasets):
            sampler_list.append(distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank))
            dataloaders.append(DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, 
                shuffle=shuffle, pin_memory=True, sampler=sampler_list[-1], prefetch_factor=2
                ))
            num_batches += len(dataloaders[-1])
            
        return dataloaders, num_batches, sampler_list
    

    def _get_num_samples(self, num_samples_per_dataset):
        sampled_datasets = []
        for dataset in self.datasets:
            print('initial', len(dataset))
            dataset_size = len(dataset)
            if num_samples_per_dataset > 0:
                indices = random.choices(range(dataset_size), k=num_samples_per_dataset)
                sampled_datasets.append(Subset(dataset, indices))
            print('after', len(sampled_datasets[-1]))
        if num_samples_per_dataset > 0:
            self.datasets = sampled_datasets
            
        num_samples = 0
        for dataset in self.datasets:
            num_samples += len(dataset)
            print(len(dataset))
        return num_samples


class EvaluationDataset:
    
    def __init__(self, args, image_size=224, image_processor=None, text_processor=None, split='test', 
                 shuffle=False, num_samples_per_dataset=400):

        self.datasets = [
            NightDataset(root_dir=args.train_data, split=split, load_size=image_size, image_processor=image_processor),
            #BAPPSDataset(root_dir=args.train_data, split=split, load_size=image_size, image_processor=image_processor, ratio=1.0), 
            HPDv2Pairs(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, split=split,
                     instruct=None),  
            HQEdit(data_dir=args.train_data, img_prepr=image_processor, text_prepr=text_processor, 
                       split=split, task_type='text_2afc'), 
            KADID10kPairs(root_dir=args.train_data, transform_fn=image_processor, sampling_mode='sev-noref', split=split),
        ]
        self.dataloader_types = ['2AFC', 'Text-Images-AFC', 'Text-2AFC', 'IQA']
        self.num_samples = num_samples_per_dataset * len(self.dataloader_types)
        self.dataloaders, self.num_batches, self.sampler_list = self._get_dataloader(batch_size=args.val_batch_size, 
                                                                                     num_workers=args.workers, shuffle=shuffle, 
                                                                                     world_size=args.world_size, rank=args.rank,
                                                                                     num_samples_per_dataset=num_samples_per_dataset)
        
        # self.dataloader_types = ['2afc', '3afc', 'text-images-afc', 'text-2afc', 'iqa']
        
    def _get_dataloader(self, batch_size, num_workers, shuffle, world_size, rank, num_samples_per_dataset):
        dataloaders, sampler_list = list(), list()
        num_batches = 0
        for idx, dataset in enumerate(self.datasets):
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            sampled_indices = random.sample(indices, num_samples_per_dataset)
            subset_dataset = Subset(dataset, sampled_indices)
            dataloaders.append(DataLoader(subset_dataset, batch_size=batch_size, num_workers=num_workers, 
                                                shuffle=shuffle, sampler=None, pin_memory=True,
                                                prefetch_factor=2))
            num_batches += len(dataloaders[-1])
            
        return dataloaders, num_batches, sampler_list
    

