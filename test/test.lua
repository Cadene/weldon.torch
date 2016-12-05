require 'nn'
require 'WeldonPooling'

local m = nn.WeldonPooling(1, 5, 2)
m:forward(torch.ones(10,5,10,10))
