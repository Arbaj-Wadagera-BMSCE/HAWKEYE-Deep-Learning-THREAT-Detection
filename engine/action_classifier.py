# engine/action_classifier.py
import torch
import cv2
import numpy as np
from config import ACTION_MODEL_NAME, CUSTOM_ACTION_MODEL_PATH, CUSTOM_ACTION_LABELS_PATH
import os
import json

# Static list of the 400 Kinetics action labels (fallback)
KINETICS_LABELS = [
    "abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery", "arm wrestling", "arranging flowers",
    "assembling computer", "auctioning", "baby waking up", "baking cookies", "balancing on a slackline", "bandaging", "barbequing",
    "bartending", "base jumping", "bathing dog", "baton twirling", "beach combing", "bee keeping", "bell ringing", "bench pressing",
    "bending back", "bending metal", "biking through snow", "blasting sand", "blowdrying hair", "blowing bubble gum", "blowing leaves",
    "blowing nose", "blowing out candles", "bobsledding", "body surfing", "bookbinding", "bottling", "bouncing on trampoline",
    "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "breaking boards", "breathing fire", "brush painting",
    "brushing hair", "brushing teeth", "building cabinet", "building lego", "building sandcastle", "building shed", "bungee jumping",
    "busking", "calculating", "calligraphy", "canoeing or kayaking", "capoeira", "capsizing", "card stacking", "card throwing",
    "carrying baby", "cartwheeling", "carving pumpkin", "casting fishing line", "catching fish", "catching or throwing baseball",
    "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "changing wheel",
    "checking tires", "cheerleading", "chewing gum", "chiseling stone", "chiseling wood", "chopping wood", "clapping",
    "clay pottery making", "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes",
    "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking egg",
    "cooking on campfire", "cooking sausages", "counting money", "country line dancing", "cracking neck", "cracking knuckles",
    "crane diving", "crawling baby", "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple",
    "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting",
    "decorating tree", "delivering mail", "digging", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics",
    "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car",
    "driving tractor", "drop kicking", "drumming fingers", "dumpster diving", "dunking basketball", "dying hair", "eating burger",
    "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti",
    "eating watermelon", "egg hunting", "embroidering", "entering car", "exercising arm", "exercising with an exercise ball",
    "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats", "fencing (sport)", "fidgeting",
    "filling eyebrows", "filling gas tank", "finger snapping", "fixing hair", "flapping wings", "flipping pancake", "flying kite",
    "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "garbage collecting", "gargling",
    "geocaching", "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving",
    "golf putting", "gospel singing in church", "grooming dog", "grooming horse", "grinding meat", "guitar strumming",
    "gymnastics tumbling", "hammer throw", "hand washing", "headbanging", "headbutting", "high jump", "high fiving", "hitching ride",
    "hockey stop", "holding snake", "home roasting coffee", "hopscotch", "horse riding", "hula hooping", "hugging", "hurdling",
    "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw", "jet skiing", "jogging", "juggling balls",
    "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball",
    "kissing", "kitesurfing", "knitting", "krumping", "land sailing", "laughing", "laying bricks", "long jump", "luge",
    "making a cake", "making a sandwich", "making bed", "making jewelry", "making pizza", "making snowman", "making sushi",
    "making tea", "marching", "massaging back", "massaging feet", "massaging legs", "massaging neck", "massaging person's head",
    "milking cow", "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "opening bottle", "opening present",
    "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)",
    "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "ping-pong", "pinching",
    "pirouetting", "planing wood", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes",
    "playing basketball", "playing bass guitar", "playing bayaiian slack key guitar", "playing beer pong", "playing billiards",
    "playing blackjack", "playing cards", "playing cello", "playing checkers", "playing chess", "playing clarinet",
    "playing controller", "playing cymbals", "playing didgeridoo", "playing dominoes", "playing drums", "playing dumbek",
    "playing flute", "playing french horn", "playing guitar", "playing hand clapping games", "playing harmonica", "playing harp",
    "playing ice hockey", "playing keyboard", "playing kickball", "playing laser tag", "playing lute", "playing mahjong",
    "playing maracas", "playing marbles", "playing monopoly", "playing oboe", "playing ocarina", "playing organ",
    "playing paintball", "playing pan pipes", "playing piano", "playing poker", "playing polo", "playing recorder",
    "playing rock-paper-scissors", "playing saxophone", "playing scrabble", "playing shuffleboard", "playing sitar",
    "playing slot machine", "playing squash or racquetball", "playing steel drums", "playing table football", "playing tambourine",
    "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball",
    "playing with trains", "playing xylophone", "pole vault", "popping balloons", "pouring beer", "pouring wine",
    "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)",
    "push up", "pushing car", "pushing cart", "pushing wheelchair", "putting on makeup", "putting on shoes", "rafting",
    "rappelling", "reading book", "reading newspaper", "recording music", "repairing phone", "ripping paper", "river tubing",
    "rock climbing", "rock scissors paper", "roller skating", "rolling pastry", "rope skipping", "rowing boat",
    "running on treadmill", "sailing", "salsa dancing", "sanding floor", "sanding wood", "sausage making", "sawing wood",
    "scaling fish", "scrambling eggs", "scrapbooking", "scuba diving", "sculpting", "separating eggs", "serving food",
    "setting table", "sewing", "shaking hands", "shaking head", "shaping clay", "shaving head", "shaving legs",
    "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put", "shoveling snow",
    "shredding paper", "shuffling cards", "side kick", "sign language interpreting", "singing", "sipping cup",
    "skateboarding", "skiing (cross-country)", "skiing (freestyle)", "skiing (slalom)", "skipping rope", "skydiving",
    "slacklining", "slapping", "sled dog racing", "sleeping", "slicing onion", "smoking", "smoking hookah",
    "snatch weight lifting", "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling",
    "somersaulting", "sorting books", "spinning poi", "spray painting", "spraying", "springboard diving", "square dancing",
    "squat", "squeezing orange", "stacking cups", "staring", "steer roping", "steering car", "sticking tongue out",
    "stomping", "stretching arm", "stretching leg", "stringing beads", "sumo wrestling", "surfing water",
    "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swimming front crawl",
    "swing dancing", "swinging legs", "swinging on something", "sword fighting", "sword swallowing", "t-shirt folding",
    "tai chi", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food",
    "tasting wine", "testifying", "texting", "threading needle", "throwing axe", "throwing discus", "tickling",
    "tie dying", "tightrope walking", "tiptoeing", "tobogganing", "tossing coin", "tossing salad", "training dog",
    "trapezing", "trimming or shaving beard", "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a tie)",
    "tying necktie", "tying shoes", "unboxing", "uncorking champagne", "unloading truck", "using computer",
    "using remote controller (not gaming)", "using segway", "vault", "waiting in line", "walking the dog", "washing dishes",
    "washing feet", "washing hair", "washing hands", "watching tv", "water skiing", "water sliding", "watering plants",
    "waving hand", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding",
    "whipping", "whistling", "windsurfing", "winking", "woodworking", "wrapping present", "wrestling", "writing",
    "yawning", "yo-yoing", "zumba"
]

class ActionClassifier:
    """Wrapper for the PyTorchVideo action recognition model."""
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] ActionClassifier using device: {self.device}")
        
        # Check if our custom-trained model exists
        if os.path.exists(CUSTOM_ACTION_MODEL_PATH) and os.path.exists(CUSTOM_ACTION_LABELS_PATH):
            print(f"--- Loading CUSTOM-TRAINED action model from {CUSTOM_ACTION_MODEL_PATH} ---")
            # We load the entire model object directly
            self.model = torch.load(CUSTOM_ACTION_MODEL_PATH, map_location=self.device)
            
            # Load our custom labels
            print(f"--- Loading CUSTOM labels from {CUSTOM_ACTION_LABELS_PATH} ---")
            with open(CUSTOM_ACTION_LABELS_PATH, 'r') as f:
                # Load the label map { "0": "fighting", "1": "neutral" }
                id_to_label_map = json.load(f)
                # Convert to a list of labels sorted by index
                # We must ensure the keys (indices) are sorted numerically
                sorted_indices = sorted([int(k) for k in id_to_label_map.keys()])
                self.custom_labels = [id_to_label_map[str(i)] for i in sorted_indices]
                
            print(f"Loaded custom labels: {self.custom_labels}")
            
        else:
            print(f"--- Loading GENERIC pre-trained action model ({ACTION_MODEL_NAME}) ---")
            print("To use a custom model, train one using train_action_classifier.py")
            # Force load from main branch to avoid compatibility issues with older torch
            self.model = torch.hub.load('facebookresearch/pytorchvideo:main', ACTION_MODEL_NAME, pretrained=True)
            self.custom_labels = KINETICS_LABELS # Use the full 400 list as fallback

        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def classify(self, clip_frames):
        """
        Classifies the action in a clip of frames.
        clip_frames: A list of numpy arrays (frames from OpenCV).
        """
        if not clip_frames:
            return None, 0.0

        # Preprocess the clip
        try:
            inputs = self._preprocess(clip_frames).to(self.device)

            # Get predictions
            with torch.no_grad():
                preds = self.model(inputs)

            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            
            pred_class_idx = preds.topk(k=1).indices[0]
            idx = pred_class_idx.item()
            
            # Handle index out of bounds if labels don't match model output
            if idx >= len(self.custom_labels):
                 print(f"[ERROR] Model prediction ({idx}) is out of bounds for label list (len: {len(self.custom_labels)})")
                 return "unknown", 0.0
                 
            pred_label = self.custom_labels[idx]
            confidence = preds[0][idx].item()
            
            return pred_label, confidence
            
        except Exception as e:
            print(f"[ActionClassifier] Error during inference: {e}")
            return "error", 0.0

    def _preprocess(self, frames):
        """Transforms a list of OpenCV frames into the required model input format."""
        # Using standard Kinetics-400 normalization stats
        mean = np.array([0.43216, 0.394666, 0.37645])
        std = np.array([0.22803, 0.22145, 0.216989])
        
        processed_frames = []
        for frame in frames:
            # Resize to a standard size expected by SlowFast (usually 256 on short side)
            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            processed_frames.append(frame)
        
        # Convert to tensor: (Channels, Time, Height, Width)
        # Stack frames -> (T, H, W, C) -> Permute to (C, T, H, W)
        tensor = torch.from_numpy(np.stack(processed_frames)).permute(3, 0, 1, 2)
        
        # Add batch dimension -> (1, C, T, H, W)
        return tensor.unsqueeze(0)