
import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import time
import warnings
warnings.filterwarnings("ignore")
import pickle as pkl

from tensorboardX import SummaryWriter

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES
else:
    raise NotImplementedError

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        t0 = time.time()
        speaker.train(interval)   # Train interval iters
        t1 = time.time()

        print()
        print("Iter: {}, time: {:.3f}".format(idx, t1 - t0))

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_unseen': {"accu": 0., "state":"", 'update':False}}
    if args.fast_train:
        log_every = 40
    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        t0 = time.time()
        # Train for log_every interval
        if aug_env is None:     # The default training process
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)   # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval//2): #back_transition
                listner.zero_grad()
                listner.env = train_env
                args.ml_weight = 0.2
                listner.accumulate_gradient(feedback_method)
                
                listner.env = aug_env
                args.ml_weight = 0.6
                listner.accumulate_gradient(feedback_method, speaker=speaker)
                listner.optim_step()

                utils.print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)
        t1 = time.time()
        print('iter: {}, time: {:.3f}'.format(idx, t1 - t0))

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1) # total actions / interval / args.batch_size
        length = max(len(listner.logs['critic_loss']), 1) # max action length / interval
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += "{:<11s} : ".format(env_name)
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ' %s: %.3f' % (metric, val)
            loss_str += '\n'

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d \n %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        print(('%s (%d %d%%)' % (timeSince(start, float(iter)/n_iters),
                                                iter, float(iter)/n_iters*100)))
        print(loss_str)

        if iter%2000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    np.random.seed(233)
    torch.manual_seed(233)
    torch.cuda.manual_seed(233)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)
    with open('./img_features/objects/pano_object_class.pkl', 'rb') as f:
        obj_dict = pkl.load(f)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, obj_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['train', 'val_seen', 'val_unseen']
    if args.submit:
        val_env_names = ['val_seen','val_unseen','test']
    else:
        pass

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, obj_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features)
    with open('./img_features/objects/pano_object_class.pkl', 'rb') as f:
        obj_dict = pkl.load(f)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, obj_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, obj_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, obj_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

