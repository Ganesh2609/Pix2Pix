import torch
from tqdm import tqdm
import matplotlib.pyplot as plt 


def train_generator(generator:torch.nn.Module,
                    discriminator:torch.nn.Module,
                    batch:tuple,
                    bce_loss:torch.nn.Module,
                    l1_loss:torch.nn.Module,
                    l1_lambda:float,
                    optimizer:torch.optim.Optimizer,
                    scaler:torch.cuda.amp.GradScaler,
                    device:torch.device):
    
    x, y = batch['Input'], batch['Target']
    x, y = x.to(device), y.to(device)
    
    generator.train()
    discriminator.eval()
    
    with torch.cuda.amp.autocast(enabled=False):
        y_fake = generator(x)
        fake_probs = discriminator(x, y_fake)
        fake_labels = torch.ones_like(fake_probs)
        loss_1 = bce_loss(fake_probs, fake_labels)
        loss_2 = l1_loss(y_fake, y)*l1_lambda
        loss = loss_1 + loss_2
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    del x, y, y_fake, fake_probs, fake_labels, loss_1, loss_2
    return loss.item()


def train_discriminator(generator:torch.nn.Module,
                        discriminator:torch.nn.Module,
                        batch:tuple,
                        bce_loss:torch.nn.Module,
                        optimizer:torch.optim.Optimizer,
                        scaler:torch.cuda.amp.GradScaler,
                        device:torch.device
                        ):
    
    x, y = batch['Input'], batch['Target']
    x, y = x.to(device), y.to(device)
    
    generator.eval()
    discriminator.train()
    
    with torch.cuda.amp.autocast(enabled=False):
        y_fake = generator(x)
        real_probs = discriminator(x, y)
        fake_probs = discriminator(x, y_fake)
        real_labels = torch.ones_like(real_probs)
        fake_labels = torch.zeros_like(fake_probs)
        loss_1 = bce_loss(real_probs, real_labels)
        loss_2 = bce_loss(fake_probs, fake_labels)
        loss = (loss_1 + loss_2)/2
    
        real_score = torch.mean(torch.sigmoid(real_probs)).item()
        fake_score = torch.mean(torch.sigmoid(fake_probs)).item()
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del x, y, y_fake, real_probs, fake_probs, real_labels, fake_labels, loss_1, loss_2
    return loss.item(), real_score, fake_score


def train_models(generator:torch.nn.Module,
                discriminator:torch.nn.Module,
                dataset:torch.utils.data.Dataset,
                dataloader:torch.utils.data.DataLoader,
                bce_loss:torch.nn.Module,
                l1_loss:torch.nn.Module,
                l1_lambda:float,
                g_optimizer:torch.optim.Optimizer,
                d_optimizer:torch.optim.Optimizer,
                g_scaler:torch.cuda.amp.GradScaler,
                d_scaler:torch.cuda.amp.GradScaler,
                device:torch.device,
                NUM_EPOCHS:int,
                g_path:str=None,
                d_path:str=None,
                result_path:str=None
                ):
    
    results = {
        'Generator Loss' : [],
        'Discriminator Loss' : []
    }
    
    for epoch in range(1, NUM_EPOCHS+1):
        
        gen_loss = 0
        disc_loss = 0
        
        j = 3
        
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            
            for i, batch in t:    
            
                disc_batch_loss, real_score, fake_score = train_discriminator(generator=generator, 
                                                                              discriminator=discriminator,
                                                                              batch=batch,
                                                                              bce_loss=bce_loss,
                                                                              optimizer=d_optimizer,
                                                                              scaler=d_scaler,
                                                                              device=device)

                gen_batch_loss = train_generator(generator=generator,
                                                 discriminator=discriminator,
                                                 batch=batch,
                                                 bce_loss=bce_loss,
                                                 l1_loss=l1_loss,
                                                 l1_lambda=l1_lambda,
                                                 optimizer=g_optimizer,
                                                 scaler=g_scaler,
                                                 device=device)

                gen_loss += gen_batch_loss
                disc_loss += disc_batch_loss
                
                t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}]')
                t.set_postfix({
                    'Gen batch loss' : gen_batch_loss,
                    'Gen loss' : gen_loss/(i+1),
                    'Disc batch loss' : disc_batch_loss,
                    'Disc loss' : disc_loss/(i+1),
                    'Real' : real_score,
                    'Fake' : fake_score
                })
                
                if i % 500 == 0:
                    results['Generator Loss'].append(gen_loss/(i+1))
                    results['Discriminator Loss'].append(disc_loss/(i+1))
                
                if g_path:
                    torch.save(obj=generator.state_dict(), f=g_path)
                
                if d_path:
                    torch.save(obj=discriminator.state_dict(), f=d_path)
                    
                if i % 100 == 0 and result_path:
                    # RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}_{j}.png'
                    RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}.png'
                    generator.eval()
                    
                    fig, axis = plt.subplots(3, 5, figsize=(16, 9))
                    
                    with torch.inference_mode():
                        x, y = batch['Input'].to(device), batch['Target'].to(device)
                        y_fake = generator(x)
                        y = torch.clamp((y+1)/2, 0, 1)
                        y_fake = torch.clamp((y_fake+1)/2, 0, 1)
                            
                    for k in range(5):
                    
                        ax = axis.flat[k]
                        ax.imshow(x[k].permute(1,2,0).cpu(), cmap='gray')
                        ax.set_title('Input')
                        ax.axis(False)
                        
                        ax = axis.flat[k+5]
                        ax.imshow(y[k].permute(1,2,0).cpu())
                        ax.set_title('Target')
                        ax.axis(False)
                        
                        ax = axis.flat[k+10]
                        ax.imshow(y_fake[k].permute(1,2,0).cpu())
                        ax.set_title('Predicted')
                        ax.axis(False);
                    
                    # plt.tight_layout()
                    plt.savefig(RESULT_SAVE_NAME)
                    plt.close(fig)
                    
                if i % 1000 == 0:
                    j+=1
                    
    return results
                        
                    