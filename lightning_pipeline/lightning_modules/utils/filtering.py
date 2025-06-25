import numpy as np
from copy import deepcopy


class AnnotFiltering:
    def __init__(self, size_thr: float = 3.0, mapping_list: list = [], is_pred: bool = False, skip_empty: bool = True):
        self.size_thr = size_thr        
        self.caregories = sum(mapping_list, [])
        self.is_pred = is_pred
        self.skip_empty = skip_empty
        
        self.mapping_dict = {old_id: new_id 
                                 for new_id, old_ids in enumerate(mapping_list) 
                                 for old_id in old_ids} if len(mapping_list) != 0 else None   # old ids to new ids
        
    def __call__(self,  data_list: list[dict]):
        if self.mapping_dict is not None:
            data_list = self.filter_map_categories(data_list)
        return self.filter_size(data_list)
    
    def mapping(self, label_list: list):
        '''Map labels to new ids'''
        return list(map(
                self.mapping_dict.get,
                label_list
            ))

    def filter_map_categories(self, data_list: list[dict]):
        '''Filter caregories and map them to new ids'''
        data_list_filtered = []
        skipped_count = 0

        for image_dict in data_list:
            label_list = np.array(deepcopy(image_dict['label']))

            mask = np.isin(label_list, self.caregories)
            
            image_dict['box'] = np.array(image_dict['box'])[mask].tolist()
            # filter and map labels to new ids
            image_dict['label'] = self.mapping(label_list[mask])
            if self.is_pred:
                image_dict['label_scores'] = np.array(image_dict['label_scores'])[mask].tolist()

            # we cant skip the predicted image if there are no bboxes left
            if (mask.sum() > 0) or (not self.skip_empty):    
                data_list_filtered.append(image_dict)
            else:
                skipped_count += 1
                continue

        if skipped_count > 0:
            print(f'Skipped {skipped_count} images while filtering caregories')

        return data_list_filtered

    def filter_size(self, data_list: list[dict]):
        '''Filter bboxes by size'''
        data_list_filtered = []
        skipped_count = 0

        for image_dict in data_list:
            bboxes = np.array(image_dict['box']) 
            spacing = np.array(image_dict['spacing'])

            # GT bboxes reconstructed as boxes circumscribed around the circle drawn on 2 labeled points as diameter
            # we'll calculate the box size as the mean of its width and height
            
            size = np.abs(np.mean(bboxes[:, 3:5] + 1 - bboxes[:, 0:2], axis=1)) * spacing[0] if len(bboxes) > 0 else np.array([])
            mask = size >= self.size_thr

            if self.is_pred:
                scores = np.array(image_dict['label_scores'])
                image_dict['label_scores'] = scores[mask].tolist()
            
            image_dict['box'] = bboxes[mask].tolist()
            image_dict['label'] = np.array(image_dict['label'])[mask].tolist()
            

            # we cant skip the predicted image if there are no bboxes left
            if (mask.sum() > 0) or (not self.skip_empty):
                data_list_filtered.append(image_dict)
            else:
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f'Skipped {skipped_count} images while filtering nudules >= {self.size_thr} mm')

        return data_list_filtered