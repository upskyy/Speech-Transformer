from model_builder import build_transformer
import torch

batch_size = 4
seq_length = 200
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

inputs = torch.FloatTensor(batch_size, input_size, seq_length).to(device)
input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30]).to(device)

output = transformer.recognize(inputs, input_lengths)
print(output.size())  # torch.Size([4, 200])
