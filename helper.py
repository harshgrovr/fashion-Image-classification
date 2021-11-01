import gzip
import os
import random
import urllib.request
import webbrowser
from collections import defaultdict

import numpy as np

from jina import Document
from jina.helper import colored
from jina.logging.predefined import default_logger
from jina.logging.profile import ProgressBar
from numpy import load
result_html = []
top_k = 0
num_docs_evaluated = 0
evaluation_value = defaultdict(float)
classes_label = {'0': 'boston_bull', '1': 'dingo', '2': 'pekinese', '3': 'bluetick', '4': 'golden_retriever', '5': 'bedlington_terrier', '6': 'borzoi', '7': 'basenji', '8': 'scottish_deerhound', '9': 'shetland_sheepdog', '10': 'walker_hound', '11': 'maltese_dog', '12': 'norfolk_terrier', '13': 'african_hunting_dog', '14': 'wire-haired_fox_terrier', '15': 'redbone', '16': 'lakeland_terrier', '17': 'boxer', '18': 'doberman', '19': 'otterhound', '20': 'standard_schnauzer', '21': 'irish_water_spaniel', '22': 'black-and-tan_coonhound', '23': 'cairn', '24': 'affenpinscher', '25': 'labrador_retriever', '26': 'ibizan_hound', '27': 'english_setter', '28': 'weimaraner', '29': 'giant_schnauzer', '30': 'groenendael', '31': 'dhole', '32': 'toy_poodle', '33': 'border_terrier', '34': 'tibetan_terrier', '35': 'norwegian_elkhound', '36': 'shih-tzu', '37': 'irish_terrier', '38': 'kuvasz', '39': 'german_shepherd', '40': 'greater_swiss_mountain_dog', '41': 'basset', '42': 'australian_terrier', '43': 'schipperke', '44': 'rhodesian_ridgeback', '45': 'irish_setter', '46': 'appenzeller', '47': 'bloodhound', '48': 'samoyed', '49': 'miniature_schnauzer', '50': 'brittany_spaniel', '51': 'kelpie', '52': 'papillon', '53': 'border_collie', '54': 'entlebucher', '55': 'collie', '56': 'malamute', '57': 'welsh_springer_spaniel', '58': 'chihuahua', '59': 'saluki', '60': 'pug', '61': 'malinois', '62': 'komondor', '63': 'airedale', '64': 'leonberg', '65': 'mexican_hairless', '66': 'bull_mastiff', '67': 'bernese_mountain_dog', '68': 'american_staffordshire_terrier', '69': 'lhasa', '70': 'cardigan', '71': 'italian_greyhound', '72': 'clumber', '73': 'scotch_terrier', '74': 'afghan_hound', '75': 'old_english_sheepdog', '76': 'saint_bernard', '77': 'miniature_pinscher', '78': 'eskimo_dog', '79': 'irish_wolfhound', '80': 'brabancon_griffon', '81': 'toy_terrier', '82': 'chow', '83': 'flat-coated_retriever', '84': 'norwich_terrier', '85': 'soft-coated_wheaten_terrier', '86': 'staffordshire_bullterrier', '87': 'english_foxhound', '88': 'gordon_setter', '89': 'siberian_husky', '90': 'newfoundland', '91': 'briard', '92': 'chesapeake_bay_retriever', '93': 'dandie_dinmont', '94': 'great_pyrenees', '95': 'beagle', '96': 'vizsla', '97': 'west_highland_white_terrier', '98': 'kerry_blue_terrier', '99': 'whippet', '100': 'sealyham_terrier', '101': 'standard_poodle', '102': 'keeshond', '103': 'japanese_spaniel', '104': 'miniature_poodle', '105': 'pomeranian', '106': 'curly-coated_retriever', '107': 'yorkshire_terrier', '108': 'pembroke', '109': 'great_dane', '110': 'blenheim_spaniel', '111': 'silky_terrier', '112': 'sussex_spaniel', '113': 'german_short-haired_pointer', '114': 'french_bulldog', '115': 'bouvier_des_flandres', '116': 'tibetan_mastiff', '117': 'english_springer', '118': 'cocker_spaniel', '119': 'rottweiler'}

def _get_groundtruths(target, pseudo_match=True):
    # group doc_ids by their labels
    a = np.squeeze(target['index-labels']['data'])
    a = np.stack([a, np.arange(len(a))], axis=1)
    a = a[a[:, 0].argsort()]
    lbl_group = np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])

    # each label has one groundtruth, i.e. all docs that have the same label are considered as matches
    groundtruths = {lbl: Document() for lbl in range(120)}
    for lbl, doc_ids in enumerate(lbl_group):
        if not pseudo_match:
            # full-match, each doc has 6K matches
            for doc_id in doc_ids:
                match = Document()
                match.tags['id'] = int(doc_id)
                groundtruths[lbl].matches.append(match)
        else:
            # pseudo-match, each doc has only one match, but this match's id is a list of 6k elements
            match = Document()
            match.tags['id'] = doc_ids.tolist()
            groundtruths[lbl].matches.append(match)

    return groundtruths


def index_generator(num_docs: int, target: dict):
    """
    Generate the index data.

    :param num_docs: Number of documents to be indexed.
    :param target: Dictionary which stores the data paths
    :yields: index data
    """
    for internal_doc_id in range(num_docs):
        # x_blackwhite.shape is (28,28)
        # x_blackwhite = 255 - target['index']['data'][internal_doc_id]
        # x_color.shape is (28,28,3)
        x_color = target['index']['data'][internal_doc_id]
        label = int(target['index-labels']['data'][internal_doc_id][0])
        # x_color = np.stack((x_blackwhite,) * 3, axis=-1)
        d = Document(content=x_color, tags={'label': label})        
        d.tags['id'] = internal_doc_id        
        yield d


def query_generator(num_docs: int, target: dict, with_groundtruth: bool = True):
    """
    Generate the query data.

    :param num_docs: Number of documents to be queried
    :param target: Dictionary which stores the data paths
    :param with_groundtruth: True if want to include labels into query data
    :yields: query data
    """
    gts = _get_groundtruths(target)
    for _ in range(num_docs):
        num_data = len(target['query-labels']['data'])
        idx = random.randint(0, num_data - 1)
        # x_blackwhite.shape is (28,28)
        # x_blackwhite = 255 - target['query']['data'][idx]
        # x_color.shape is (28,28,3)
        # x_color = np.stack((x_blackwhite,) * 3, axis=-1)
        x_color = target['query']['data'][idx]
        
        d = Document(content=x_color) 
        if with_groundtruth:
            gt = gts[target['query-labels']['data'][idx][0]]            
            yield d, gt
        else:
            yield d


def print_result(resp):
    """
    Callback function to receive results.

    :param resp: returned response with data
    """
    global top_k
    global evaluation_value
    global classes_label 
    
    for d in resp.docs:
        vi = d.uri
        result_html.append(f'<tr><td><img src="{vi}"/></td><td>')
        result_html.append(f'<label for="{classes_label[d.id]}"/> {classes_label[d.id]}</label>')
        result_html.append(f'<span style="display:inline-block; width: 20px;"></span>')
        top_k = len(d.matches)
        for kk in d.matches:
            kmi = kk.uri
            result_html.append(
                f'<img src="{kmi}" style="opacity:{kk.scores["cosine"].value}"/>'
            )
        result_html.append('</td></tr>\n')

        # update evaluation values
        # as evaluator set to return running avg, here we can simply replace the value
        for k, evaluation in d.evaluations.items():
            evaluation_value[k] = evaluation.value


def write_html(html_path):
    """
    Method to present results in browser.

    :param html_path: path of the written html
    """

    with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'demo.html')
    ) as fp, open(html_path, 'w') as fw:
        t = fp.read()
        t = t.replace('{% RESULT %}', '\n'.join(result_html))
        t = t.replace(
            '{% PRECISION_EVALUATION %}',
            '{:.2f}%'.format(evaluation_value['Precision'] * 100.0),
        )
        t = t.replace(
            '{% RECALL_EVALUATION %}',
            '{:.2f}%'.format(evaluation_value['Recall'] * 100.0),
        )
        t = t.replace('{% TOP_K %}', str(top_k))
        fw.write(t)

    url_html_path = 'file://' + os.path.abspath(html_path)

    try:
        webbrowser.open(url_html_path, new=2)
    except:
        pass  # intentional pass, browser support isn't cross-platform
    finally:
        default_logger.info(
            f'You should see a "demo.html" opened in your browser, '
            f'if not you may open {url_html_path} manually'
        )

    colored_url = colored(
        'https://github.com/jina-ai/jina', color='cyan', attrs='underline'
    )
    default_logger.info(
        f'🤩 Intrigued? Play with `jina hello fashion --help` and learn more about Jina at {colored_url}'
    )


def download_data(targets, download_proxy=None, task_name='download fashion-mnist'):
    """
    Download data.

    :param targets: target path for data.
    :param download_proxy: download proxy (e.g. 'http', 'https')
    :param task_name: name of the task
    """
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # if download_proxy:
    #     proxy = urllib.request.ProxyHandler(
    #         {'http': download_proxy, 'https': download_proxy}
    #     )
    #     opener.add_handler(proxy)
    # urllib.request.install_opener(opener)
    # with ProgressBar(description=task_name) as t:
    for k, v in targets.items():
        # if not os.path.exists(v['filename']):
        #     urllib.request.urlretrieve(
        #         v['url'], v['filename'], reporthook=lambda *x: t.update(0.01)
        #     )

        if k == 'index-labels' or k == 'query-labels':
            v['data'] = load_labels(v['filename'])
        if k == 'index' or k == 'query':
            v['data'] = load_mnist(v['filename'])


def load_mnist(path):
    """
    Load MNIST data

    :param path: path of data
    :return: MNIST data in np.array
    """

    # load numpy array from npz file
    
    # load dict of arrays
    data = load(path)['data'].reshape([-1, 80, 80, 3])
    return data

    # with gzip.open(path, 'rb') as fp:
    #     return np.frombuffer(fp.read(), dtype=np.uint8, offset=16).reshape([-1, 28, 28])


def load_labels(path: str):
    """
    Load labels from path

    :param path: path of labels
    :return: labels in np.array
    """
    
    # load dict of arrays
    data = load(path)['data'].reshape([-1, 1])    
    return data

    # with gzip.open(path, 'rb') as fp:
    #     return np.frombuffer(fp.read(), dtype=np.uint8, offset=8).reshape([-1, 1])
