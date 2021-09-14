from transformers import BertTokenizer
from character_bert.modeling.character_bert import CharacterBertModel
from character_bert.utils.character_cnn import CharacterIndexer
import unittest

class TestCharacterBERT(unittest.TestCase):

    def test_tokenizer(self):
        '''
        Test both loading the tokenizer and whether it works
        '''
        
        # Example text
        x = "Hello World!"

        # Tokenize the text
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.assertIsNotNone(x, msg='Successfully loaded bert-base-uncased tokenizer')

        x = tokenizer.basic_tokenizer.tokenize(x)
        self.assertEqual(len(x), 3, msg='Tokenizer successfully worked')
    
    def test_general_character_bert_works(self):
        '''
        Test that the general Character BERT model loads and returns the correct embedding shape. 
        '''
        
        # Test data
        x = "Hello World!"

        # Load model
        model = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/')
        self.assertIsNotNone(model, 'Successfully loaded the general character bert model')

        # Add [CLS] and [SEP]
        x = ['[CLS]', *x, '[SEP]']

        # Convert token sequence into character indices
        indexer = CharacterIndexer()

        batch = [x]  # This is a batch with a single token sequence x
        batch_ids = indexer.as_padded_tensor(batch)

        # Feed batch to CharacterBERT & get the embeddings
        embeddings_for_batch, _ = model(batch_ids)
        embeddings_for_x = embeddings_for_batch[0]
        self.assertIsNotNone(embeddings_for_x, msg='Successfully made general character bert embeddings')
        self.assertEqual(list(embeddings_for_x.size()), [14, 768])
    
    def test_medical_character_bert_works(self):
        '''
        Test that the medical Character BERT model loads and returns the correct embedding shape. 
        '''

        # Test data
        x = "Hello World!"

        # Load model
        model = CharacterBertModel.from_pretrained('./pretrained-models/medical_character_bert/')
        self.assertIsNotNone(model, 'Successfully loaded the medical character bert model')

        # Add [CLS] and [SEP]
        x = ['[CLS]', *x, '[SEP]']

        # Convert token sequence into character indices
        indexer = CharacterIndexer()
        batch = [x]  # This is a batch with a single token sequence x
        batch_ids = indexer.as_padded_tensor(batch)

        # Feed batch to CharacterBERT & get the embeddings
        embeddings_for_batch, _ = model(batch_ids)
        embeddings_for_x = embeddings_for_batch[0]
        self.assertIsNotNone(embeddings_for_x, msg='Successfully made medical character bert embeddings')
        self.assertEqual(list(embeddings_for_x.size()), [14, 768])

if __name__ == '__main__':
    unittest.main()
