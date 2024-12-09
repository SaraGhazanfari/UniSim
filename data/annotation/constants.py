answer_templates = {
    
    '2afc':[
      "The image that is more similar to the reference image is image {label}.",
      "Image {label} is the one that most closely resembles the reference image.",
      "Out of the two, image {label} is more similar to the reference image.",
      "The image that best matches the reference image is image {label}.",
      "Compared to the reference image, image {label} shows a greater similarity.",
      "Image {label} is the most similar to the reference image.",
      "When comparing to the reference image, image {label} is the closer match.",
      "The reference image is more closely aligned with image {label}.",
      "Of the two options, image {label} is more like the reference image.",
      "Between the two, image {label} is more alike the reference image.",
      "Image {label}", "Image {label}", "Image {label}", "Image {label}", "Image {label}", 
      "Image {label}", "Image {label}", "Image {label}", "Image {label}", "Image {label}", 
      ],

    'text_images_afc': [
        "The image that is more similar to the reference caption is image {label}.",
        "Image {label} is the one that most closely aligns the reference caption.",
        "Out of the two, image {label} is more representative of the reference caption.",
        "The image that best matches the reference caption is image {label}.",
        "Compared to the reference caption, image {label} shows a greater alignment.",
        "Image {label} is the best illustration of the reference caption.",
        "When comparing to the reference caption, image {label} is the closer match.",
        "The reference caption is more closely aligned with image {label}.",
        "Of the two options, image {label} is more depictive of the reference caption.",
        "Between the two, image {label} is more alike the reference caption.",
        "Image {label}", "Image {label}", "Image {label}", "Image {label}", "Image {label}", 
        "Image {label}", "Image {label}", "Image {label}", "Image {label}", "Image {label}", 
        ],

    'text_2afc': [
        "The caption that is more aligned to the reference image is caption {label}.",
        "Caption {label} is the one that aligns more closely with the reference image.",
        "The caption that best matches the reference image is caption {label}.",
        "Of the two, caption {label} matches more with the reference image.",
        "Caption {label} is better aligned with the reference image.",
        "When compared to the reference image, caption {label} is a closer match.",
        "Caption {label} is the most aligned with the reference image.",
        "Between the two, caption {label} is more descriptive of the reference image.",
        "Out of the captions, caption {label} shows a stronger alignment with the reference image.",
        "Caption {label} corresponds more closely to the reference image.",
        "Caption {label}", "Caption {label}", "Caption {label}", "Caption {label}", "Caption {label}", 
        "Caption {label}", "Caption {label}", "Caption {label}", "Caption {label}", "Caption {label}", 
        
        ],

    'iqa': [
        "The image with better quality is image {label}.",
        "Image {label} has superior quality.",
        "Out of the two, image {label} offers better quality.",
        "The image with higher quality is image {label}.",
        "Image {label} is of higher quality.",
        "Of the two, image {label} has the best quality.",
        "Between the two, image {label} is of better quality.",
        "Image {label}", "Image {label}", "Image {label}", "Image {label}", "Image {label}", 
        "Image {label}", "Image {label}", "Image {label}", "Image {label}", "Image {label}", 
        ]
    }

instruction_templates = {
    
    '2afc': {
        'Image1': [   
            'Answer the following question:\nHere are three images: <image> <image> <image>. '\
            'If image 1 is the reference image, which image of the other two is more similar to the reference image?',
            
            'Answer the following multiple-choice question:\nIf <image> is the reference image, '\
            'which image of the other two images <image> <image> is more similar to the reference image? '\
            '(Assume reference image is image 1 and the other two are image 2 and image 3)'\
            '\nOptions:\n(A) Image 3\n(B) Image 2',
            
            'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. '\
            'If image 1 is the reference image, which image of the other two is more similar to the reference image?'\
            '\nOptions:\n(A) Image 2\n(B) Image 3',
            
            'Answer the following question:\nIf <image> is the reference image, '\
            'which image of the other two images <image> <image> is more similar to the reference image?',
            
            'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. '\
            'If image 1 is the reference image, which image of the other two is more similar to the reference image?'\
            '\nOptions:\n(A) Image 3\n(B) Image 2',
            
            'Answer the following question:\nIf image 1 is the reference image, which image of the other two is more similar '\
            'to the reference image? <image> <image> <image>. '\
            '(Assume reference image is image 1 and the other two are image 2 and image 3)',
            
            'Answer the following multiple-choice question:\nIf image 1 is the reference image, '\
            'which image of the other two is more similar to the reference image? '\
            '<image> <image> <image>.\nOptions:\n(A) Image 2\n(B) Image 3',
            
            'Answer the following multiple-choice question:\nIf <image> is the reference image, which image of '\
            'the other two images <image> <image> is more similar to the reference image? '\
            '(Assume reference image is image 1 and the other two are image 2 and image 3)\nOptions:\n(A) Image 2\n(B) Image 3',
            ],
        
        'Image2': [
            'Answer the following question:\nHere are three images: <image> <image> <image>. '\
            'If image 2 is the reference image, which image of the other two is more similar to the reference image?',
            
            'Answer the following multiple-choice question:\nIf image 2 is the reference image, '\
            'which image of the other two is more similar to the reference image? <image> <image> <image>.'\
            '\nOptions:\n(A) Image 1\n(B) Image 3',
            
            'Answer the following question:\nIf image 2 is the reference image, '\
            'which image of the other two (Image 1 or Image 3) is more similar to the reference image?'\
            '<image> <image> <image>.',
            
            'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. '\
            'If image 2 is the reference image, which image of the other two is more similar to the reference image?'\
            '\nOptions:\n(A) Image 1\n(B) Image 3',
            
            'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. '\
            'If image 2 is the reference image, which image of the other two is more similar to the reference image?'\
            '\nOptions:\n(A) Image 3\n(B) Image 1',
            
            'Answer the following multiple-choice question:\nIf image 2 is the reference image, '\
            'which image of the other two is more similar to the reference image? <image> <image> <image>.'
            '\nOptions:\n(A) Image 3\n(B) Image 1',
        ],    
        
        'Image3': [            
            'Answer the following question:\nHere are three images: <image> <image> <image>. '\
            'If image 3 is the reference image, which image of the other two is more similar to the reference image?',
            
            'Answer the following multiple-choice question:\nIf image 3 is the reference image, '
            'which image of the other two is more similar to the reference image? '\
            '<image> <image> <image>.\nOptions:\n(A) Image 1\n(B) Image 2',
            
            'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. '\
            'If image 3 is the reference image, which image of the other two is more similar to the reference image?'\
            '\nOptions:\n(A) Image 2\n(B) Image 1',
            
            'Answer the following multiple-choice question:\nHere are three images: <image> <image> <image>. '\
            'If image 3 is the reference image, which image of the other two is more similar to the reference image?'\
            '\nOptions:\n(A) Image 1\n(B) Image 2',
            
            'Answer the following multiple-choice question:\nIf image 3 is the reference image, which image of the other two is'\
            'more similar to the reference image? <image> <image> <image>.\nOptions:\n(A) Image 2\n(B) Image 1',
            
            'Answer the following question:\nIf image 3 is the reference image, which image of the other two is more similar to the reference image? '\
            '<image> <image> <image>. (Assume reference image is image 3 and the other two are image 1 and image 2)',
            ],        
    },
    'text_images_afc': ['Answer the following question:\nHere are two images: <image> <image>, '\
                        'and here is the reference caption: "{caption}". which of the two images is '\
                        'more aligned to the reference caption?',
                        
                        'Answer the following question:\nHere are two images: <image> <image>, '\
                        'and here is the reference caption: "{caption}". which of the two images is '\
                        'a better match for the reference caption?',
                        
                        'Answer the following multiple-choice question:\nIf caption: "{caption}" is the reference caption, '\
                        'which image of the two images <image> <image> is more similar to the reference caption? ' \
                        '\nOptions:\n(A) Image 1\n(B) Image 2',
                        
                        'Answer the following multiple-choice question:\nHere are two images: <image> <image>. '\
                        'If caption: "{caption}" is the reference caption, which image of the two is more similar to the reference caption?'
                        '\nOptions:\n(A) Image 2\n(B) Image 1',
                        
                        'Answer the following question:\nIf caption: "{caption}" is the reference caption, '\
                        'which image of the two images <image> <image> is more aligned with the reference caption?',
                        
                        'Answer the following multiple-choice question:\nHere are two images: <image> <image>. '\
                        'If caption: "{caption}" is the reference caption, which image better matches to the reference caption?'\
                        '\nOptions:\n(A) Image 2\n(B) Image 1',
                        
                        'Answer the following question:\nIf caption: "{caption}" is the reference caption, ' \
                        'which image of the two is the closer match to the reference caption? '\
                        '<image> <image>.',
                        
                        'Answer the following question:\nIf "{caption}" is the reference caption, '\
                        'which image of the two is a better depiction of the reference caption? '\
                        '<image> <image>. (Assume the two images are image 1 and image 2)',
                        
                        'Answer the following multiple-choice question:\nIf "{caption}" is the reference caption, '\
                        'which image of the two is more closely aligned with the reference caption? '\
                        '<image> <image>.\nOptions:\n(A) Image 1\n(B) Image 2',
                        
                        'Answer the following multiple-choice question:\nIf caption: "{caption}" is the reference caption, '\
                        'which image of the other two images <image> <image> more matches to the reference caption? '
                        'Options:\n(A) Image 1\n(B) Image 2'],
    
    
    
    'text_2afc': ['Answer the following question:\nHere is the reference image: <image>. '\
                'Here are two captions, caption 1: "{caption1}", and caption 2: "{caption2}". '\
                'which of the two captions is the one that aligns more closely with the reference image?',
                  
                'Answer the following question:\nHere are two captions, caption 1: "{caption1}", and caption 2: "{caption2}". '\
                'If <image> the reference image, which caption of the two is more similar to the reference image?',
                
                'Answer the following multiple-choice question:\nIf <image> is the reference image, '\
                'which caption of the two captions, caption 1: "{caption1}", and caption 2: "{caption2}" best matches to the reference image? '\
                '\nOptions:\n(A) Caption 2\n(B) Caption 1',
                
                'Answer the following multiple-choice question:\nHere are two captions, caption 1: "{caption1}", and caption 2: "{caption2}". '\
                'If <image> is the reference image, which caption of the two captions matches more with the reference image?' \
                '\nOptions:\n(A) Caption 1\n(B) Caption 2',
                
                'Answer the following question:\nIf <image> is the reference image, '\
                'which of the two captions, caption 1: "{caption1}", and caption 2: "{caption2}" '\
                'is the most aligned with the reference image?',
                
                'Answer the following multiple-choice question:\nHere are two captions, caption 1: "{caption1}", and caption 2: "{caption2}". '\
                'If <image> is the reference image, which caption of the two is more descriptive of the reference image?'\
                '\nOptions:\n(A) Caption 2\n(B) Caption 1',
                
                'Answer the following question:\nIf <image> is the reference image, '\
                'which caption of the two shows a stronger alignment with the reference image? '\
                'caption 1: "{caption1}", and caption 2: "{caption2}"',
                
                'Answer the following question:\nIf <image> is the reference image, '\
                'which caption of the two corresponds more closely to the reference image? '\
                'caption 1: "{caption1}", and caption 2: "{caption2}"',
                
                'Answer the following multiple-choice question:\nIf <image> is the reference image, '\
                'which caption of the two is more aligned to the reference image? '\
                'caption 1: "{caption1}", and caption 2: "{caption2}".\nOptions:\n(A) Caption 1\n(B) Caption 2',
                
                'Answer the following multiple-choice question:\nIf <image> is the reference image, which image of '\
                'the other two captions, caption 1: "{caption1}", and caption 2: "{caption2}" is the best match to the reference image? '\
                '\nOptions:\n(A) Caption 1\n(B) Caption 2'],
    
    'iqa': ['Answer the following question:\nHere are two images: <image> <image>. which of the two images has a better quality?',
            
            'Answer the following question:\nHere are two images: <image> <image>. '\
            'which of the two images has a better quality?',
            
            'Answer the following multiple-choice question:\nwhich image of the two images <image> <image>'
            ' has a better quality? \nOptions:\n(A) Image 2\n(B) Image 1',
            
            'Answer the following multiple-choice question:\nHere are two images: <image> <image>. '\
            'which image of the two  which of the two images has a better quality?\nOptions:\n(A) Image 1\n(B) Image 2',
            
            'Answer the following question:\nwhich of the two images has a better quality? '\
            'Here are the two images <image> <image>.',
            
            'Answer the following multiple-choice question:\nHere are two images: <image> <image>. '\
            ' which of the two images has a better quality?\nOptions:\n(A) Image 2\n(B) Image 1',
            
            'Answer the following question:\n which of the two images has a better quality? '\
            '<image> <image>. \nOptions:\n(A) Image 2\n(B) Image 1',
        
            'Answer the following question:\n which of the two images has a better quality? '\
            '<image> <image>.',
            
            'Answer the following multiple-choice question:\nwhich image of the two  which of the two images <image> <image> has a better quality? '\
            '\nOptions:\n(A) Image 1\n(B) Image 2',
           ]
}
