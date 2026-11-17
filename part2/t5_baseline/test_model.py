"""Small CPU integration test. Run in the Part 2 environment; no downloads needed.

This checks the model/loss/checkpoint mechanics on a randomly initialized toy T5.
It is not an evaluation of pretrained T5-small or of text-to-SQL accuracy.
"""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from transformers import T5Config, T5ForConditionalGeneration
from train_baseline import save_checkpoint, restore_state, make_loader


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
                      num_layers=1, num_decoder_layers=1, num_heads=2,
                      pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
                      dropout_rate=0.0)
    model = T5ForConditionalGeneration(config)
    batch = {'input_ids': torch.tensor([[3,4,1,0],[5,6,7,1]]),
             'attention_mask': torch.tensor([[1,1,1,0],[1,1,1,1]]),
             'labels': torch.tensor([[8,9,1,-100],[10,11,12,1]])}
    shifted = model.prepare_decoder_input_ids_from_labels(batch['labels'])
    assert shifted.tolist() == [[0,8,9,1],[0,10,11,12]]
    model.eval()
    first_loss = model(**batch).loss
    padded = {**batch, 'labels': torch.nn.functional.pad(batch['labels'], (0,2), value=-100)}
    assert torch.allclose(first_loss, model(**padded).loss, atol=1e-6)

    # A token-weighted accumulation window should match the full-batch gradient.
    full = copy.deepcopy(model)
    split = copy.deepcopy(model)
    full(**batch).loss.backward()
    for i in range(2):
        micro = {k:v[i:i+1] for k,v in batch.items()}
        n = (micro['labels'] != -100).sum()
        (split(**micro).loss * n / 7).backward()
    for a,b in zip(full.parameters(), split.parameters()):
        assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
    assert torch.isfinite(loss) and model(**batch).loss < first_loss
    predicted = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                               max_new_tokens=5)
    assert predicted.shape[0] == 2
    # Dummy tokenizer avoids any remote download during the checkpoint test.
    fake_tokenizer = SimpleNamespace(save_pretrained=lambda path: (Path(path)/'toy.txt').write_text('toy'))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'last'
        save_checkpoint(path, model, fake_tokenizer, optimizer, scheduler, {'epoch':1})
        expected_rng = torch.rand(3)
        loaded = T5ForConditionalGeneration.from_pretrained(path)
        loaded.eval()
        new_opt = torch.optim.AdamW(loaded.parameters(), lr=0.01)
        new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt, lambda _:1)
        assert restore_state(path,new_opt,new_sched)['epoch'] == 1
        assert torch.equal(torch.rand(3),expected_rng)
        assert torch.allclose(loaded(**batch).logits,model(**batch).logits,atol=1e-6)
        assert new_sched.last_epoch == scheduler.last_epoch
        # Verify the actual optimizer moments survive, not just the model weights.
        for old,new in zip(optimizer.state.values(),new_opt.state.values()):
            assert torch.equal(old['exp_avg'],new['exp_avg'])
        save_checkpoint(path, model, fake_tokenizer, optimizer, scheduler, {'epoch':2})
        assert (path/'training_state.pt').exists()
        assert not (Path(tmp)/'last.previous').exists()
    features = [{'x':i} for i in range(20)]
    collator = lambda items: [item['x'] for item in items]
    a = list(make_loader(features,collator,3,42,epoch=2))
    b = list(make_loader(features,collator,3,42,epoch=2))
    assert a == b and sorted(x for batch in a for x in batch) == list(range(20))
    print('PASS: decoder shift, loss masking, accumulation, forward/backward, generation, checkpoint reload, RNG, sampler.')


if __name__ == '__main__':
    main()
