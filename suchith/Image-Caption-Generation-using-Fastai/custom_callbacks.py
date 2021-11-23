
class TeacherForcingCallback(Callback):
    def __init__(self, learn:Learner, decay_epochs=3):
        super().__init__()
        self.learn = learn
        self.decay_iterations = decay_epochs * len(self.learn.data.train_ds) // self.learn.data.batch_size
    
    def on_batch_begin(self, iteration,**kwargs):
        self.learn.model.decoder.teacher_forcing_ratio = (self.decay_iterations-iteration) * 1/self.decay_iterations if iteration < self.decay_iterations else 0
        
    def on_batch_end(self,**kwargs):
        self.learn.model.decoder.teacher_forcing_ratio = 0.

class GradientClipping(LearnerCallback):
    "Gradient clipping during training."
    def __init__(self, learn:Learner, clip:float = 0.3):
        super().__init__(learn)
        self.clip = clip

    def on_backward_end(self, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip: nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)

        

class BleuMetric(Callback):
    def on_epoch_begin(self, **kwargs):
        self.references = list()
        self.candidates = list()
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        pred, decode_lengths,_,inds = last_output
        references = metadata.references[metadata.index.isin(inds.tolist())]
        _,pred_words = pred.max(dim=-1)
        vocab_i2s = dict(zip(vocab.values(),vocab.keys()))
        hypotheses = list()
        for cap in pred_words: hypotheses.append([vocab_i2s[x] for x in cap.tolist() if x not in 
                               {vocab['<start>'], vocab['<end>'], vocab['<pad>']}])
        self.references.extend(references)
        self.candidates.extend(hypotheses)

        
    def on_epoch_end(self, last_metrics, **kwargs):
        assert len(self.references) == len(self.candidates)
        return add_metrics(last_metrics, corpus_bleu(self.references, self.candidates, weights=(0.5,0.5,0.5,0.5)))
