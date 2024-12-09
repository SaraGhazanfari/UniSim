import json


def test_annotation(ann_list):
    for ann in ann_list:
        correct_label = ann['raw_label']
        has_option = False
        qs = ann['conversations'][0]['value']
        answer = ann['conversations'][1]['value']
        try:
            # assert qs.count('which of') <= 1
            assert qs.count('which image of') <= 1
            assert qs.count('which caption of') <= 1
            if 'Options' in qs and '(A)' in qs:
                has_option = True
            if ann['datasource'] == '2afc':
                assert qs.count("reference caption") == 0
                assert qs.count("reference image") >= 1
                assert qs.count("<image>") == 3
                
                if has_option:
                    qs = qs.split('\n')[-2:]
                    options = [int(qs[-2].split('Image')[-1].strip()), int(qs[-1].split('Image')[-1].strip())]
                    if options[0] > options[1]:
                        assert answer == ['A', 'B'][1-correct_label]
                    else:
                        assert answer == ['A', 'B'][correct_label]
                else:
                    if 'reference image is image 1' in qs.lower() or 'image 1 is the reference' in qs.lower():
                        assert ['image 2', 'image 3'][correct_label] in answer.lower() 
                    elif 'reference image is image 2' in qs.lower() or 'image 2 is the reference' in qs.lower():
                        assert ['image 1', 'image 3'][correct_label] in answer.lower() 
                    elif 'reference image is image 3' in qs.lower() or 'image 3 is the reference' in qs.lower():
                        assert ['image 1', 'image 2'][correct_label] in answer.lower() 
                    
                    
            elif ann['datasource'] == 'text_images_afc':
                assert qs.count("<image>") == 2
                assert qs.count("reference caption") >= 1
                # assert qs.count("reference image") == 0
                if has_option:
                    qs = qs.split('\n')[-2:]
                    options = [int(qs[-2].split('Image')[-1].strip()), int(qs[-1].split('Image')[-1].strip())]
                    if options[0] > options[1]:
                        assert answer == ['A', 'B'][1-correct_label]
                    else:
                        assert answer == ['A', 'B'][correct_label]
                else:
                    assert ['image 1', 'image 2'][correct_label] in answer.lower() 

                    
                    
            elif ann['datasource'] =='text_2afc':
                assert qs.count("<image>") == 1
                assert qs.count("reference caption") == 0
                assert qs.count("reference image") >= 0
                
                if has_option:
                    qs = qs.lower().split('\n')[-2:]
                    options = [int(qs[-2].split('caption')[-1].strip()), int(qs[-1].split('caption')[-1].strip())]
                    if options[0] > options[1]:
                        assert answer == ['A', 'B'][1-correct_label]
                    else:
                        assert answer == ['A', 'B'][correct_label]
                else:
                    assert ['caption 1', 'caption 2'][correct_label] in answer.lower() 

                    
            elif ann['datasource'] == 'iqa':
                assert qs.count("<image>") == 2
                assert qs.count("reference image") == 0
                assert qs.count("reference caption") == 0
                if has_option:
                    qs = qs.split('\n')[-2:]
                    options = [int(qs[-2].split('Image')[-1].strip()), int(qs[-1].split('Image')[-1].strip())]
                    if options[0] > options[1]:
                        assert answer == ['A', 'B'][1-correct_label]
                    else:
                        assert answer == ['A', 'B'][correct_label]
                else:
                    assert ['image 1', 'image 2'][correct_label] in answer.lower()
                    
        except Exception as e:
            print(e)
            print(ann['datasource'], ann)   
            raise e

if __name__ == "__main__":
    path = '/scratch/sg7457/code/lmm_unisim/playground/night_unisim_instruct_annotation.json'
    with open(path) as file:
        ann_list = json.load(file)
        
    test_annotation(ann_list)
