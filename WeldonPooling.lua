-- compatible CUDA
require 'torch'
require 'nn'

local WeldonPooling, parent = torch.class('nn.WeldonPooling', 'nn.Module')

-- n: number of top instances
function WeldonPooling:__init(nMax, nMin)
    parent.__init(self)
    self.nMax = nMax
    self.nMin = nMin or nMax

    self.output = torch.Tensor()
    self.indicesMax = torch.Tensor()
    self.indicesMin = torch.Tensor()
end

function WeldonPooling:updateOutput(input)
    -- backward compatibility

    if input:dim() == 4 then -- batch
        local batchSize = input:size(1)
        local numChannels = input:size(2)
        local h = input:size(3)
        local w = input:size(4)

        local nMax = self.nMax
        local nMin = self.nMin

        self.output:typeAs(input):resize(batchSize, numChannels, 1, 1)
        local x = input:view(batchSize, numChannels, h*w)

        -- sort scores by decreasing order
        local scoreSorted, indices = torch.sort(x, x:size():size(), true)

        -- compute top max
        self.indicesMax = indices[{{},{},{1,nMax}}]
        torch.sum(self.output, scoreSorted[{{},{},{1,nMax}}], 3)
        self.output:div(nMax)

        -- compute top min
        if nMin > 0 then
            self.indicesMin = indices[{{},{},{h*w-nMin+1,h*w}}]
            local yMin = torch.sum(scoreSorted[{{},{},{h*w-nMin+1,h*w}}], 3):div(nMin)
            torch.add(self.output, self.output, yMin)
        end

        self.output = self.output:view(batchSize, numChannels, 1, 1)

    else
        print('error in WeldonPooling:updateOutput')
    end
    return self.output
end

function WeldonPooling:updateGradInput(input, gradOutput)

    if input:dim() == 4 then -- batch

        local batchSize = input:size(1)
        local numChannels = input:size(2)
        local h = input:size(3)
        local w = input:size(4)

        local nMax = self.nMax
        local nMin = self.nMin

        local yMax = torch.expand(gradOutput:clone():view(batchSize, numChannels, 1), batchSize, numChannels, nMax)
        local z = torch.zeros(batchSize, numChannels, h*w):typeAs(input)
        z:scatter(3, self.indicesMax, yMax):div(nMax)

        if nMin > 0 then
            local yMin = torch.expand(gradOutput:clone():view(batchSize, numChannels, 1):div(nMin), batchSize, numChannels, nMin)
            self.gradInput = z:scatter(3, self.indicesMin, yMin):view(batchSize, numChannels, h, w)
        else
            self.gradInput = z
        end

    else
        print('error in WeldonPooling:updateGradInput')
    end
    return self.gradInput
end

function WeldonPooling:empty()
    self.gradInput:resize()
    self.gradInput:storage():resize(0)
    self.output:resize()
    self.output:storage():resize(0)
    self.indicesMax:resize()
    self.indicesMax:storage():resize(0)
    self.indicesMin:resize()
    self.indicesMin:storage():resize(0)
end

function WeldonPooling:__tostring__()
    local s =  string.format('%s(nMax=%d,nMin=%d)', torch.type(self), self.nMax, self.nMin)
    return s
end
