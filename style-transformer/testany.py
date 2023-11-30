import torch
from models import StyleTransformer, Discriminator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop

class Config():
    data_path = './data/yelp/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 16
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0

best_ckpt = 8850
ckpt_folder = "save/Nov07025149"

def main():
    # print(torch.__version__)
    # print(torch.cuda.is_available())

    config = Config()
    # train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    # torch.save(vocab, 'save/vocab_obj.pth')

    vocab = torch.load('save/vocab_obj.pth')

    # print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)

    model_F.load_state_dict(torch.load(ckpt_folder + '/ckpts/' + str(best_ckpt) + '_F.pth'))
    model_D.load_state_dict(torch.load(ckpt_folder + '/ckpts/' + str(best_ckpt) + '_D.pth'))

    model_F.eval()
    model_D.eval()
    eos_idx = vocab.stoi['<eos>']

    while(True):

        # inp_str = "i 'd definitely recommend giving them a try ."
        inp_str = input("Enter a sentence: ")

        inp_token = [vocab.stoi[s] for s in inp_str.split()] + [eos_idx]
        inp_token = inp_token[:16]
        inp_length = [len(inp_token)]
        inp_token = inp_token + [1]*(16-inp_length[0])

        inp_token = torch.tensor(inp_token).view(1, -1).to(config.device)
        inp_length = torch.tensor(inp_length).to(config.device)

        with torch.no_grad():
            style = model_D(inp_token, inp_length)

            style = 1 - torch.argmax(style[:,1:], 1)

        print("Style is", "positive" if style[0].cpu() == 0 else "negative" )

        with torch.no_grad():
            raw_log_probs = model_F(
                inp_token,
                None,
                inp_length,
                style,
                generate=True,
                differentiable_decode=False,
            )

        rev_sentence =  tensor2text(vocab, raw_log_probs.argmax(-1).cpu())

        print("Rev sentence: ", rev_sentence[0])
    

if __name__ == '__main__':
    main()
