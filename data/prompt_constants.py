DATASET_TYPE = {'night': '2afc',
                'bapps': '2afc',
                'pie-app-2afc': '2afc',

                'things': '3afc',
                'cifar100': '3afc',
                'cifar100coarse': '3afc',
                'cifar100coarse-triplets': '3afc',
                'imagenet-triplets': '3afc',
                'r-imagenet-triplets': '3afc',
                'bal-r-imagenet-triplets': '3afc',
                'h-imagenet-triplets': '3afc',

                'imagereward1k-pairs': 'text_images_afc',
                'hpdv2-pairs': 'text_images_afc',
                'hpdv2-easy-pairs': 'text_images_afc',
                'hpdv2-mindist-pairs': 'text_images_afc',
                'hq_edit_text_images': 'text_images_afc',
                'spot_the_diff': 'text_images_afc',
                'magic_brush': 'text_images_afc',
                'birds_to_words': 'text_images_afc',
                'agiqa3k-align-pairs': 'text_images_afc',

                'hpdv2-3a': 'text_images_3a',
                'agiqa3k-align-3a': 'text_images_many',

                'kadid-pairs': 'iqa',  # image quality assessment
                'easy-kadid-pairs': 'iqa',
                'kadid-corr-pairs': 'iqa',
                'koniq-pairs': 'iqa',
                'pipal': 'iqa',
                'agiqa3k-qual-pairs': 'iqa',
                'pie-app-iqa': 'iqa',

                'koniq-colorfulness-pairs': 'colorfulness',
                'koniq-brightness-pairs': 'brightness',
                'koniq-contrast-pairs': 'contrast',
                'koniq-sharpness-pairs': 'sharpness',
                'sice-pairs': 'brightness',
                'sice-pairs-h': 'brightness',

                'foil': 'text_2afc',
                'cub-pairs': 'text_2afc',
                'coco-triplets': 'text_2afc',
                'coco-gen-triplets': 'text_2afc',
                'h-coco-triplets': 'text_2afc',
                'h-coco-gen-triplets': 'text_2afc',
                'hq_edit_text_2afc': 'text_2afc',
                'polaris': 'text_2afc',
                }

llava_next_prompt = {
    '2afc': 'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. ' \
            'If image 1 is the reference image, which image of the other two is more similar to the reference image?' \
            '\nOptions:\n(A) Image 2\n(B) Image 3',

    '3afc': "Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>" \
            "Which one (A, B, C) is the odd-one-out of the group?" \
            '\nOptions:\n(A) Image 1\n(B) Image 2\n(C)Image 3',

    'text_images_afc': 'Answer the following question:\nHere are two images: <image> <image>, ' \
                       'and here is the reference caption: "{prompt}". which of the two images is ' \
                       'more aligned to the reference caption?' \
                       '\nOptions:\n(A) Image 1\n(B) Image 2',

    'text_images_3a': 'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>, ' \
                      'and here is the reference caption: "{prompt}". which of the three images is ' \
                      'more aligned to the reference caption?',

    'iqa': 'Answer the following multiple-choice question:\nGiven two images: image 1 <image> and image 2 <image>.' \
           'which image has a better quality?\n' \
           '\nOptions:\n(A) Image 1\n(B) Image 2',

    'text_2afc': 'Answer the following multiple-choice question:\n Given the reference image: <image>. ' \
                 'and two captions, caption 1: "{cap1}", caption 2: "{cap2}" \n' \
                 'which caption has a better alignment with the reference image?  \n' \
                 '\nOptions:\n(A) Caption 1\n(B) Caption 2',

    'colorfulness': 'Answer the following multiple-choice question:\nGiven two images: image 1 <image> and image 2 <image>.' \
                    ' which image is more colorful?' \
                    '\nOptions:\n(A) Image 1\n(B) Image 2',

    'brightness': 'Answer the following multiple-choice question:\nGiven two images: image 1 <image> and image 2 <image>.' \
                  ' which image is brighter?' \
                  '\nOptions:\n(A) Image 1\n(B) Image 2',

    'contrast': 'Answer the following multiple-choice question:\nGiven two images: image 1 <image> and image 2 <image>.' \
                ' which image has more contrast?' \
                '\nOptions:\n(A) Image 1\n(B) Image 2',

    'sharpness': 'Answer the following multiple-choice question:\nGiven two images: image 1 <image> and image 2 <image>.' \
                 ' which image is sharper in color?' \
                 '\nOptions:\n(A) Image 1\n(B) Image 2',
}

llava_prompt = {'2afc': "Take the middle image as reference. Can you tell which one of the left or right image is most" \
                        "similar to the center one? Select from the following choices.(A) left (B) right",
                '3afc': '',
                'text_images_afc': '',
                'iqa': None,
                'text_2afc': None,
                'colorfulness': None,
                'brightness': None,
                'contrast': None,
                'sharpness': None,
                }

mantis_prompt = {'2afc': f"Image 1 is the reference image\n" \
                         f"which one of the Image 2 or Image 3 is most similar to the reference image? \n" \
                         f"select from the following choices: (A) Image 2 (B) Image 3",

                 '3afc': 'Which one Image 1, Image 2 or Image 3 is the odd-one-out of the group?',
                 'text_images_afc': '''Given the following text: {prompt}
                                        which of the Image 1 or Image 2 is more aligned with the given text? 
                                        select from the following choices: (A) Image 1 (B) Image 2''',
                 'iqa': '''which of the Image 1 or Image 2 has a better quality?
                        select from the following choices: (A) Image 1 (B) Image 2''',
                 'text_2afc': '''Given two captions, caption 1: {cap1}, caption 2: {cap2}
                                 which caption has a better alignment with the Image 1? 
                                 select from the following choices: (A) caption 1 (B) caption 2''',
                 'colorfulness': '''which of the Image 1 or Image 2 is more colorful?
                        select from the following choices: (A) Image 1 (B) Image 2''',
                 'brightness': '''which of the Image 1 or Image 2 is brighter?
                        select from the following choices: (A) Image 1 (B) Image 2''',
                 'contrast': 'which of the Image 1 or Image 2 has more contrast?' \
                             '\nOptions:\n(A) Image 1\n(B) Image 2',
                 'sharpness': 'which of the Image 1 or Image 2 is sharper in color?' \
                              '\nOptions:\n(A) Image 1\n(B) Image 2',

                 'text_images_many': 'Here are three images, and here is the reference caption: "{prompt}". ' \
                                     'which of the Image 1, Image 2 or Image 3 is more aligned to the reference caption? \n' \
                                     '\nOptions:\n(A) Image 1\n(B) Image 2\n(C)Image 3',

                 }

compare2score_prompt = {
    '2afc': f"<image> <image> Compared with the first image, what is your similarity rating for second image?",
    '3afc': f"<image> <image> Compared with the first image, what is your similarity rating for second image?",

    'text_images_afc': '''Given the following text: {prompt}
                                        and two images: image 1 <image> and image 2 <image>.
                                        which image is more aligned with the given text?
                                        select from the following choices: (A) Image 1 (B) Image 2''',
    'iqa': "<image> <image> Compared with the first image, what is your quality rating for second image?",
    'text_2afc': '''Given the image: <image>.
                                     and two captions, caption 1: {cap1}, caption 2: {cap2}
                                     which caption has a better alignment with the image?
                                     select from the following choices: (A) caption 1 (B) caption 2''',
    'colorfulness': '''Given two images: image 1 <image> and image 2 <image>.
                               which image is more colorful?
                               select from the following choices: (A) Image 1 (B) Image 2''',
    'brightness': '''Given two images: image 1 <image> and image 2 <image>.
                               which image is more brighter?
                               select from the following choices: (A) Image 1 (B) Image 2''',
    'contrast': '',  # TODO: add this.
    'sharpness': '',  # TODO: add this.
}

PROMPT_DICT = {'llava': llava_prompt,
               'llava-next': llava_next_prompt,
               'mantis': mantis_prompt,
               'compare-to-score': compare2score_prompt}

LONG_STRING = """A quite long random prompt, it does not need to be task-specfic, just to fill the tokenizer max number of tokens.
Apparently lower quality images are closer to any text embedding, which is quite surprising.
Maybe this is due to the modalities gap: low quality images are out-of-distribution respect to training images, then their embedding are far away from those of the uncorrupted images.
This should be verified on other datasets."""

ATTRIBUTE_PROMPTS = {
    'clip-iqa': {  # From https://arxiv.org/abs/2207.12396
        'quality': ('Good photo.', 'Bad photo.'),
        'colorfulness': ('Colorful photo.', 'Dull photo'),
        'brightness': ('Bright photo.', 'Dark photo.'),
        'contrast': ('High contrast photo.', 'Low contrast photo.'),
        'sharpness': ('Sharp photo.', 'Blurry photo.')
    },
    'naive': {
        'quality': 'A high-quality photo.',
        'colorfulness': 'A very colorful photo.',
        'brightness': 'A bright photo.',
        'contrast': 'A high contrast photo.',
        'sharpness': 'A very sharp photo',
    }
}


def _get_model_type(name):
    if 'llava-next' in name or 'unisim' in name:
        return 'llava-next'
    elif 'llava' in name:
        return 'llava'
    elif 'Mantis' in name:
        return 'mantis'
    elif 'Compare2Score' in name:
        return 'compare-to-score'
    else:
        return 'other'
