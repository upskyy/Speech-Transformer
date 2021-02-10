from model_builder import build_transformer
import torch
import torch.nn as nn

batch_size = 4
seq_length = 200
target_length = 10
input_size = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

transformer = build_transformer(
    device,
    num_vocabs=10,
    model_dim=16,
    ff_dim=32,
    num_layers=3,
    num_heads=2,
    dropout=0.3,
    max_len=200,
    ff_type='ff',
    pad_id=0,
    sos_id=1,
    eos_id=2,
    input_size=80,
    use_joint_ctc_attention=False,
)

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-04)

for i in range(10):
    inputs = torch.FloatTensor(batch_size, input_size, seq_length).to(device)
    input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)

    _, _, dec_output_prob, _, _, _ = transformer(inputs, input_lengths, targets)

    loss = criterion(dec_output_prob.contiguous().view(-1, dec_output_prob.size(-1)), targets[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()

    print(loss)
# tensor(2.4562, grad_fn=<NllLossBackward>)
# tensor(2.4253, grad_fn=<NllLossBackward>)
# tensor(2.4010, grad_fn=<NllLossBackward>)
# tensor(2.3064, grad_fn=<NllLossBackward>)
# tensor(2.4401, grad_fn=<NllLossBackward>)
# tensor(2.1642, grad_fn=<NllLossBackward>)
# tensor(2.1931, grad_fn=<NllLossBackward>)
# tensor(2.1918, grad_fn=<NllLossBackward>)
# tensor(2.1025, grad_fn=<NllLossBackward>)
# tensor(2.0370, grad_fn=<NllLossBackward>)

