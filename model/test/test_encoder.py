from model_builder import build_encoder
import torch
import torch.nn as nn

batch_size = 4
seq_length = 200
target_length = 10
input_size = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

transformer_encoder = build_encoder(
    num_vocabs=20,
    model_dim=16,
    ff_dim=32,
    num_layers=3,
    num_heads=2,
    dropout=0.3,
    max_len=200,
    ff_type='ff',
    input_size=80,
    use_joint_ctc_attention=True,
).to(device)

criterion = nn.CTCLoss(blank=3, zero_infinity=True)
optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=1e-04)

for i in range(10):
    inputs = torch.FloatTensor(batch_size, input_size, seq_length).to(device)
    input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7, 7])

    _, output_lengths, enc_output_probs, _ = transformer_encoder(inputs, input_lengths)

    loss = criterion(enc_output_probs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)
    loss.backward()
    optimizer.step()

    print(loss)
# tensor(33.8276, grad_fn= < MeanBackward0 >)
# tensor(28.7312, grad_fn= < MeanBackward0 >)
# tensor(27.2621, grad_fn= < MeanBackward0 >)
# tensor(26.6050, grad_fn= < MeanBackward0 >)
# tensor(25.9407, grad_fn= < MeanBackward0 >)
# tensor(25.3616, grad_fn= < MeanBackward0 >)
# tensor(25.0820, grad_fn= < MeanBackward0 >)
# tensor(24.6277, grad_fn= < MeanBackward0 >)
# tensor(24.3415, grad_fn= < MeanBackward0 >)
# tensor(23.8878, grad_fn= < MeanBackward0 >)
