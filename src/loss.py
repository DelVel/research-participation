import einops
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
    return loss


def cross_loss(img, text):
    loss = 0
    img_size = img.shape[0]
    text_size = text.shape[0]
    for i in range(img_size):
        for j in range(text_size):
            similarity = similarity_criteria(img[i], text[j])
            if i == j:
                loss += -similarity.max()
            else:
                loss += similarity.max()
    return loss


def similarity_criteria(img, text, *, reduce='sum'):
    img_s = einops.repeat(img, 'img latent -> k img latent', k=text.shape[0])
    text_s = einops.repeat(text, 'text latent -> text k latent', k=img.shape[0])
    sim_mat = cosine_similarity(img_s, text_s, -1)
    max_mat = einops.reduce(sim_mat, 'text img -> text', 'max')
    res = einops.reduce(max_mat, 'text ->', reduce)
    return res
