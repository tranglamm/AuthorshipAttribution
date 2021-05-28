"""

Download pre-trained common-crawl vectors from fastText's website
https://fasttext.cc/docs/en/crawl-vectors.html 

Save it to aa/ressources/pretraines_emb


@author: Trang Lam - github.com/tranglamm
"""

from logging import getLogger
import io
import numpy as np
import os
import numpy as np
import sys
import shutil
import gzip


from urllib.request import urlopen


valid_lang_ids = {"af", "sq", "als", "am", "ar", "an", "hy", "as", "ast",
                  "az", "ba", "eu", "bar", "be", "bn", "bh", "bpy", "bs",
                  "br", "bg", "my", "ca", "ceb", "bcl", "ce", "zh", "cv",
                  "co", "hr", "cs", "da", "dv", "nl", "pa", "arz", "eml",
                  "en", "myv", "eo", "et", "hif", "fi", "fr", "gl", "ka",
                  "de", "gom", "el", "gu", "ht", "he", "mrj", "hi", "hu",
                  "is", "io", "ilo", "id", "ia", "ga", "it", "ja", "jv",
                  "kn", "pam", "kk", "km", "ky", "ko", "ku", "ckb", "la",
                  "lv", "li", "lt", "lmo", "nds", "lb", "mk", "mai", "mg",
                  "ms", "ml", "mt", "gv", "mr", "mzn", "mhr", "min", "xmf",
                  "mwl", "mn", "nah", "nap", "ne", "new", "frr", "nso",
                  "no", "nn", "oc", "or", "os", "pfl", "ps", "fa", "pms",
                  "pl", "pt", "qu", "ro", "rm", "ru", "sah", "sa", "sc",
                  "sco", "gd", "sr", "sh", "scn", "sd", "si", "sk", "sl",
                  "so", "azb", "es", "su", "sw", "sv", "tl", "tg", "ta",
                  "tt", "te", "th", "bo", "tr", "tk", "uk", "hsb", "ur",
                  "ug", "uz", "vec", "vi", "vo", "wa", "war", "cy", "vls",
                  "fy", "pnb", "yi", "yo", "diq", "zea"}
                  
def _print_progress(downloaded_bytes, total_size):
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    sys.stdout.write(" (%0.2f%%) [" % percent)
    sys.stdout.write("=" * bar)
    sys.stdout.write(">")
    sys.stdout.write(" " * (bar_size - bar))
    sys.stdout.write("]\r")
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write('\n')

def _download_file(url, write_file_name, chunk_size=2**13):
    print("Downloading %s" % url)
    response = urlopen(url)
    if hasattr(response, 'getheader'):
        file_size = int(response.getheader('Content-Length').strip())
    else:
        file_size = int(response.info().getheader('Content-Length').strip())
    downloaded = 0
    download_file_name = write_file_name + ".part"
    with open(download_file_name, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            _print_progress(downloaded, file_size)

    os.rename(download_file_name, write_file_name)


def _download_gz_model(gz_file_name, if_exists):
    if os.path.isfile(gz_file_name):
        if if_exists == 'ignore':
            return True
        elif if_exists == 'strict':
            print("gzip File exists. Use --overwrite to download anyway.")
            return False
        elif if_exists == 'overwrite':
            pass

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/%s" % gz_file_name

    #pretrained_dir= the directory which stores fasttext embeddings = pretrained_emb
    pretrained_dir="aa/ressources/pretrained_emb" 
    gz_file_name=os.path.join(pretrained_dir,gz_file_name)
    _download_file(url, gz_file_name)

    return True


def download_model(lang_id, format_vec, if_exists='strict' ,dimension=None):
    """
        Download pre-trained common-crawl vectors from fastText's website
        https://fasttext.cc/docs/en/crawl-vectors.html
    """
    if lang_id not in valid_lang_ids:
        raise Exception("Invalid lang id. Please select among %s" %
                        repr(valid_lang_ids))

    file_name = "cc.%s.300.%s" % (lang_id,format_vec)
    gz_file_name = "%s.gz" % file_name
    pretrained_dir="aa/ressources/pretrained_emb" 
    file_name=os.path.join(pretrained_dir,file_name)

    if os.path.isfile(file_name):
        if if_exists == 'ignore':
            return file_name
        elif if_exists == 'strict':
            print("File exists. Use --overwrite to download anyway.")
            return
        elif if_exists == 'overwrite':
            pass

    if _download_gz_model(gz_file_name, if_exists):
        gz_file_name=os.path.join(pretrained_dir,gz_file_name)
        with gzip.open(gz_file_name, 'rb') as f:
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f, f_out)

    return file_name




def read_txt_embeddings(path, params):
    '''
    Reload pretrained embeddings from a text file.
    '''
    word2id = {}
    vectors = []

    # load pretrained embeddings
    #_emb_dim_file = params.emb_dim
    _emb_dim_file=300
    with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if word in word2id:
                logger.warning("Word \"%s\" found twice!" % word)
                continue
            if not vect.shape == (_emb_dim_file,):
                logger.warning("Invalid dimension (%i) for word \"%s\" in line %i."
                               % (vect.shape[0], word, i))
                continue
            assert vect.shape == (_emb_dim_file,)
            word2id[word] = len(word2id)
            vectors.append(vect[None])

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pretrained word embeddings from %s" % (len(vectors), path))

    # compute new vocabulary / embeddings
    embeddings = np.concatenate(vectors, 0)
    #embeddings = torch.from_numpy(embeddings).float()

    #assert embeddings.size() == (len(word2id), params.emb_dim)
    assert embeddings.shape == (len(word2id), params.emb_dim)

    return word2id, embeddings

"""

def load_bin_embeddings(path, params):
    '''
    Reload pretrained embeddings from a fastText binary file.
    '''
    model = load_fasttext_model(path)
    assert model.get_dimension() == params.emb_dim
    words = model.get_labels()
    logger.info("Loaded binary model from %s" % path)

    # compute new vocabulary / embeddings
    embeddings = np.concatenate([model.get_word_vector(w)[None] for w in words], 0)
    embeddings = torch.from_numpy(embeddings).float()
    word2id = {w: i for i, w in enumerate(words)}
    logger.info("Generated embeddings for %i words." % len(words))

    assert embeddings.size() == (len(word2id), params.emb_dim)
    return word2id, embeddings
"""