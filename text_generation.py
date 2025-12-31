import torch
from dataset_manager import *
from neural_networks.transformer import Transformer

# Convert a string into a tensor for the input
def str_to_tensor_fr(prompt : str) -> torch.Tensor:
    global analyzer_fr, vec_fr
    prompt = analyzer_fr(prompt)
    prompt = list(map(lambda x: vec_fr.vocabulary_.get(x), prompt))
    prompt = list(filter(lambda x : x is not None, prompt))
    prompt = torch.tensor(prompt).unsqueeze(0)
    return prompt

def str_to_tensor_en(prompt : str, device : str = None) -> torch.Tensor:
    global analyzer_en, vec_en
    prompt = analyzer_en(prompt)
    prompt = list(map(lambda x: vec_en.vocabulary_.get(x), prompt))
    prompt = torch.tensor(prompt).unsqueeze(0)
    if device is None:
        return prompt
    else:
        return prompt.to(device = device)


def generate_text(model : torch.nn.Module, prompt : str, start : str = 'sssss', print_prompt : bool = False, max_len : int = 100):
    global invert_vocabulary_en, device

    if print_prompt:
        print('💁‍♂️ :', prompt)

    # Converting to tensors
    prompt = str_to_tensor_fr(prompt)
    text_generated = str_to_tensor_en(start)

    # Putting data on appropriate device 
    prompt = prompt.to(device = device)
    text_generated = text_generated.to(device = device)

    # Start printing the text
    print('🤖 :', end = ' ')
    last_word = ''

    i = 0
    while last_word != 'eeeee' and last_word != 'vvvvv' and i < max_len:
        
        # Computing the output of the model
        output = model(prompt, text_generated)
        last_output = output[:, -1, :].squeeze().argmax()

        # Get the output word
        last_word = invert_vocabulary_en.get(last_output.item())

        if last_word == 'eeeee':
            break
        elif last_word == 'vvvvv':
            raise ValueError('A void token was generated !')
        else:
            if i == 0:
                last_word = last_word[0].upper() + last_word[1:]
                print(last_word, end = ' ', flush = True)
            else:
                print(last_word, end = ' ', flush = True)

        i += 1

        # Adding the corresponding word to the input
        text_generated = torch.cat((text_generated, str_to_tensor_en(last_word, device = device)), dim = 1)

    # Return on the line
    print()


def interact_with_user(model : torch.nn.Module):
    query = input("💁‍♂️ : ").strip() + ' eeeee'
    while query.strip() != 'eeeee':
        generate_text(model, query)
        print()
        query = input("💁‍♂️ : ").strip() + ' eeeee'

device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Getting the dataset
df = load_dataset('../data/eng-fra.txt')

# Get vectorizer for french and english
vec_fr, vec_en = make_vectorizers(df)

# Get vocabularies lengths
vocabulary_size_fr = len(vec_fr.vocabulary_)
vocabulary_size_en = len(vec_en.vocabulary_)

# Get padding tokens
VOID_TOKEN_FR = vec_fr.vocabulary_.get('vvvvv')
VOID_TOKEN_EN = vec_en.vocabulary_.get('vvvvv')

# Get their analyzers
analyzer_fr, analyzer_en = make_analyzers(vec_fr, vec_en)

# Invert vocabularies
invert_vocabulary_fr, invert_vocabulary_en = invert_vocabularies(vec_fr, vec_en)

date = '2025-12-31_12-43-03'

model_path = '/Users/samsam-dz/Documents/CoursENS/deep_learning/TP/translator/results/' + date + '/model.pth'

model = Transformer(vocabulary_size_fr, vocabulary_size_en, pad_fr = VOID_TOKEN_FR, pad_en = VOID_TOKEN_EN)
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to('mps')
model.eval()

# Start interaction
interact_with_user(model)





