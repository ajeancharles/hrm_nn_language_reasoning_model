import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import random

class SentenceEncoder(nn.Module):
    """Encodes a batch of padded sentences into a tensor of sentence vectors."""
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, sentences, lengths):
        """
        Args:
            sentences (Tensor): Padded sentences, shape (batch_size, max_len).
            lengths (Tensor): Original lengths of sentences, shape (batch_size).
        Returns:
            Tensor: Sentence vectors, shape (batch_size, hidden_size).
        """
        embedded = self.embedding(sentences)
        # Pack sequence to ignore padding during GRU computation
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed_embedded)
        return hidden.squeeze(0)

class DocumentEncoder(nn.Module):
    """Encodes a document (a sequence of sentence vectors) into a single document vector."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # This GRU processes a sequence of sentence vectors
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, sentence_vectors):
        """
        Args:
            sentence_vectors (Tensor): A tensor of sentence vectors for one document,
                                       shape (num_sentences, sentence_hidden_size).
        Returns:
            Tensor: The document vector, shape (1, doc_hidden_size).
        """
        # Add a batch dimension of 1 because we process one document at a time.
        _, hidden = self.gru(sentence_vectors.unsqueeze(0))
        return hidden.squeeze(0)

class SentenceDecoder(nn.Module):
    """Decodes a sentence given a document context vector."""
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # The input to the GRU is the concatenation of the current word's embedding
        # and the document context vector.
        self.gru = nn.GRU(embedding_dim + context_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_word, hidden, context):
        """
        Performs one step of decoding.
        Args:
            input_word (Tensor): The input word for this step, shape (batch_size, 1).
            hidden (Tensor): The hidden state from the previous step.
            context (Tensor): The document context vector.
        Returns:
            A tuple of (prediction, hidden state).
        """
        embedded = self.embedding(input_word) # (batch_size, 1, embedding_dim)
        context_expanded = context.unsqueeze(1) # (batch_size, 1, context_size)

        # Concatenate the word embedding with the context vector
        gru_input = torch.cat([embedded, context_expanded], dim=2)

        output, hidden = self.gru(gru_input, hidden)
        prediction = self.fc(output)
        return prediction, hidden

class HierarchicalModel(nn.Module):
    """The main model bringing together the encoder and decoder."""
    def __init__(self, vocab, emb_dim=64, sent_hidden=128, doc_hidden=128):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.sentence_encoder = SentenceEncoder(self.vocab_size, emb_dim, sent_hidden)
        self.document_encoder = DocumentEncoder(sent_hidden, doc_hidden)
        self.sentence_decoder = SentenceDecoder(self.vocab_size, emb_dim, sent_hidden, doc_hidden)

    def forward(self, document, loss_fn, teacher_forcing_ratio=0.5):
        """
        A forward pass for training.
        Args:
            document (list of Tensors): A list of sentence tensors for one document.
            loss_fn: The loss function (e.g., CrossEntropyLoss).
            teacher_forcing_ratio (float): The probability to use teacher forcing.
        Returns:
            The average loss for the document.
        """
        total_loss = 0
        num_sentences = len(document)

        # Step 1: Encode all sentences in the document into vectors
        sent_lengths = torch.tensor([len(s) for s in document], dtype=torch.long)
        padded_sents = rnn_utils.pad_sequence(document, batch_first=True, padding_value=self.vocab.pad_idx)
        sentence_vectors = self.sentence_encoder(padded_sents, sent_lengths)

        # Step 2: Encode the document from the sequence of sentence vectors
        doc_vector = self.document_encoder(sentence_vectors)

        # Step 3: Decode each sentence, conditioned on the document vector
        for sent_idx, sent_tensor in enumerate(document):
            target_len = len(sent_tensor)

            # The initial hidden state for the decoder is the sentence's own encoded vector
            decoder_hidden = sentence_vectors[sent_idx].unsqueeze(0).unsqueeze(0)

            use_teacher_forcing = random.random() < teacher_forcing_ratio

            sentence_loss = 0
            if use_teacher_forcing:
                # Teacher forcing: Feed the ground-truth target tokens as the next input
                decoder_inputs = sent_tensor[:-1].unsqueeze(0) # (1, seq_len-1)

                # We need to run the decoder step-by-step to feed the hidden state
                # A simpler way for teacher forcing is to process the whole sequence.
                # Let's create a loop for clarity.
                for t in range(target_len - 1):
                    input_word = sent_tensor[t].unsqueeze(0).unsqueeze(0) # (1, 1)
                    prediction, decoder_hidden = self.sentence_decoder(input_word, decoder_hidden, doc_vector)
                    # prediction is (1, 1, vocab_size), target is (1)
                    sentence_loss += loss_fn(prediction.squeeze(1), sent_tensor[t+1].unsqueeze(0))
            else:
                # No teacher forcing: Use the decoder's own prediction as the next input
                input_word = sent_tensor[0].unsqueeze(0).unsqueeze(0) # Start with <sos>
                for t in range(target_len - 1):
                    prediction, decoder_hidden = self.sentence_decoder(input_word, decoder_hidden, doc_vector)
                    sentence_loss += loss_fn(prediction.squeeze(1), sent_tensor[t+1].unsqueeze(0))

                    # Get the most likely next word (greedy decoding)
                    top1 = prediction.argmax(2)
                    input_word = top1

            total_loss += (sentence_loss / (target_len - 1))

        return total_loss / num_sentences
