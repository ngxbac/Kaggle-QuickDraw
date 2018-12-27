# Tổng hợp ý tưởng (tạm thời), mọi người cùng đóng góp và cho ý kiên
Please respect Kaggle rules
-	Only submit to test new ideas, test LB
-	Or when there are un-used submissions (do not create 2 accounts)
-	Any external dataset, pre-trained weights must be posted in the forum

# Dataset: 340 classes
- Train: unbalanced distribution
- Test: (almost) balanced distribution
Test draws were collected from a different period, different locations
=> countries are useless, time (in the train set) is difficult to exploited

## Proposed split
- Train set was already shuffled, no need to re-shuffled again
- Keep last 10K for blending – blending set, please consider this sub-set as test set, do not even use it for the validation in level 0
- Number of draws per class
snowman     340029
potato      329204
calendar    321981
...
ceiling fan    115413
bed            113862
panda          113613


# Approach
Due to the size of the data, it is fine to use blending instead of stacking
## Level 0
Please keep the model weights (and the seeds) and produce probabilities for the Test set and the blending set!
 
### GrayImage-based models
If needed, external dataset could be used here
(Bac, please comment here!)

### ColorImage-based models
If needed, external dataset could be used here

### Stroke-based models
-	LSTM
Around LB 0.87 with 75K draws/class
-	RANET (another type of LSTM)
Around LB 0.87 with 75K draws/class
-	Wavenet
Around LB 0.87 with 75K draws/class
- ConvLSTM, mỗi timestep là một bức ảnh đang được vẽ, timestep sau hoàn thiện hơn timestep trước
(idea from Hau)

## Level 1
- Feed the features from Level 0 in to XGBOOST, RF, NN (CNN)
- Extra features: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/70680
- Statistics features from raw data: # strokes, # points ...

## Level 2
Weighted average of predictions from Level 1 => One final submission
## Post-processing
Balance the distribution from Level 2 => One final submission
LB improvement: 0.005

# Label Encoder proposal
word_encoder = LabelEncoder()
word_encoder.classes_ = np.array(['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa',
       'airplane', 'alarm clock', 'ambulance', 'angel',
       'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus',
       'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball',
       'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach',
       'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle',
       'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry',
       'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain',
       'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer',
       'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator',
       'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle',
       'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan',
       'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle',
       'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer',
       'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile',
       'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher',
       'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon',
       'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow',
       'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face',
       'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace',
       'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops',
       'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog',
       'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee',
       'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer',
       'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter',
       'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse',
       'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass',
       'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail',
       'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern',
       'laptop', 'leaf', 'leg', 'light bulb', 'lighthouse', 'lightning',
       'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox',
       'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone',
       'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain',
       'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail',
       'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven',
       'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants',
       'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear',
       'peas', 'pencil', 'penguin', 'piano', 'pickup truck',
       'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
       'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
       'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
       'rainbow', 'rake', 'remote control', 'rhinoceros', 'river',
       'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
       'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver',
       'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts',
       'shovel', 'sink', 'skateboard', 'skull', 'skyscraper',
       'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel',
       'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat',
       'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel',
       'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches',
       'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean',
       'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set',
       'sword', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone',
       'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe',
       'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
       'tractor', 'traffic light', 'train', 'tree', 'triangle',
       'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van',
       'vase', 'violin', 'washing machine', 'watermelon', 'waterslide',
       'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass',
       'wristwatch', 'yoga', 'zebra', 'zigzag'], dtype=object)




