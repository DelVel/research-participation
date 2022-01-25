from einops import repeat, reduce
from torch.nn.functional import cosine_similarity


def minimize_maximum_cosine(img, text):
    loss = 0
    loss += self_pairwise_loss(img)
    loss += self_pairwise_loss(text)
    loss += cross_loss(img, text)
    return loss


def self_pairwise_loss(t):
    loss = 0
    t_shape = t.shape[0]
    for i in range(t_shape):
        for j in range(i + 1, t_shape):
            loss += similarity_criteria(t[i], t[j])
    loss /= (t_shape * (t_shape - 1)) / 2
    return loss


def cross_loss(img, text):
    assert img.shape[0] == text.shape[0]
    loss = 0
    batch_size = img.shape[0]
    for i in range(batch_size):
        for j in range(i, batch_size):
            similarity = similarity_criteria(img[i], text[j])
            if i == j:
                loss += -similarity.max()
            else:
                loss += similarity.max()
    loss /= (batch_size * (batch_size + 1)) / 2
    return loss


def similarity_criteria(img, text):
    img_s = repeat(img, 'img latent -> k img latent', k=text.shape[0])
    text_s = repeat(text, 'text latent -> text k latent', k=img.shape[0])
    sim_mat = cosine_similarity(img_s, text_s, -1)
    max_mat = reduce(sim_mat, 'text img -> text', 'max')
    res = reduce(max_mat, 'text ->', 'mean')
    return res
