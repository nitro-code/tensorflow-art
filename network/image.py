from PIL import Image, ImageOps

CROP_LEFT = 25
CROP_RIGHT = 25
CROP_BOTTOM = 25
CROP_TOP = 25

WIDTH = 512
HEIGHT = 512
CHANNELS = 3

ARTISTS = ['abrulloff', 'aivazovsky', 'alekseev', 'almatadema', 'altdorfer', 'altman', 'angelico', 'antropov', 'arcimboldo', 'arp', 'baldung', 'barth', 'bazzile', 'bellini', 'bellotto', 'bogoliubov', 'bonington', 'bonnard', 'borisov-musatov', 'borovikovsky', 'bosch', 'botticelli', 'boucher', 'boudin', 'bouguereau', 'bouts', 'braque', 'briullov', 'broederlam', 'brouwer', 'bruegel', 'bruni', 'burne-jones', 'campin', 'canaletto', 'caravaggio', 'carpaccio', 'cassatt', 'castagno', 'cezanne', 'chagall', 'chardin', 'chirico', 'christinek', 'christus', 'cima', 'cimabue', 'clouet', 'cole', 'constable', 'copley', 'corot', 'correggio', 'cranach', 'cross', 'daumier', 'david', 'degas', 'delacroix', 'denis', 'dobuzhinsky', 'domenico', 'duccio', 'duchamp', 'durer', 'elgreco', 'ernst', 'eyck', 'fabriano', 'fantin-latour', 'fedotov', 'flemishstilife', 'fouquet', 'fragonard', 'gagarin', 'gainsborough', 'gauguin', 'gay', 'gerardavid', 'ghirlandao', 'giorgione', 'giotto', 'goes', 'gonzales', 'gossaert', 'goya', 'grabar', 'grosz', 'grunewald', 'hals', 'hay', 'heemskerch', 'hemessen', 'hogarth', 'holbein', 'hooch', 'huntwh', 'ingres', 'ivanov', 'jordaens', 'kandinsky', 'kaufman', 'khrutsky', 'kiprensky', 'klee', 'korovin', 'kramskoy', 'kuinji', 'kustodiyev', 'lampi', 'landseer', 'latour', 'lawrence', 'lebedev', 'leonardo', 'levitan', 'levitzky', 'leyster', 'lippi', 'lorenzetti', 'lotto', 'magritte', 'manet', 'mantegna', 'martini', 'martynov', 'massys', 'matisse', 'matveev', 'mayr', 'memling', 'mengs', 'michelangelo', 'millais', 'miro', 'modigliani', 'monet', 'morisot', 'munter', 'murillo', 'nesterov', 'nikitin', 'orlovsky', 'ostade', 'ostroumova', 'ostrovsky', 'paterssen', 'patinir', 'perov', 'perugino', 'pesne', 'petrov-vodkin', 'picabia', 'picasso', 'piero', 'pissaro', 'pointillism', 'polenov', 'poussin', 'quarenghi', 'ramsay', 'raphael', 'ray', 'rembrandt', 'renoir', 'repin', 'reynolds', 'ribera', 'rivera', 'rokotov', 'rossetti', 'rossika', 'rubens', 'ryabushkin', 'sadovnikov', 'sargent', 'savrasov', 'schongauer', 'scorel', 'semiradsky', 'semshchedrin', 'serov', 'serusier', 'seurat', 'shchedrin', 'shibanov', 'shishkin', 'signac', 'signorelli', 'sisley', 'snyders', 'somov', 'soutine', 'steen', 'stubbs', 'surikov', 'tanguy', 'tchistyakov', 'teniers', 'terborch', 'tissot', 'titian', 'tolstoy', 'tropinin', 'turner', 'uccello', 'valckenborch', 'valloton', 'vandyck', 'vangogh', 'vasilyev', 'vasnetsov', 'velazquez', 'venetsianov', 'vereshchagin', 'vermeer', 'verocchio', 'vigeelebrun', 'vishnyakov', 'vorobiev', 'vos', 'vrubel', 'vuillard', 'watteau', 'weyden', 'whistler', 'wilkie', 'winterhalter', 'witz', 'zubov', 'zurbaran']
ARTISTS_LEN = len(ARTISTS)


def preprocess_image(image_source):
  image = Image.open(image_source)
  image = ImageOps.autocontrast(image, 0.1)
  image = image.crop((CROP_LEFT, CROP_TOP, image.width - CROP_RIGHT, image.height - CROP_BOTTOM))
  return image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
