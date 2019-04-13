#Para treino (train)

python main.py --scale_factor=2 --mode=train --num_workers=0 --model_save_step=1000 --sample_step=100 --image_size=64 \
--num_channels=3 \
--batch_size=16 \
--content_loss_function='l2' \
--total_step=61001 \
--trained_discriminator=10000_cont_L1_resid_L1_DISC__loss_gen_0.212_loss_disc_0.001.pth.tar \
--trained_model=10000_cont_L1_resid_L1_GEN__loss_gen_0.212_loss_disc_0.001.pth.tar



   # Test
    parser.add_argument('--test_mode', type=str, default='pick_from_set', choices=['single', 'many', 'pick_from_set', 'evaluate'])
    parser.add_argument('--test_image_path', type=str) #Use with a single file for 'single_test' and a folder for 'many_tests'
    parser.add_argument('--evaluation_step', type=int, default=10) #evaluation log print step
    parser.add_argument('--evaluation_size', type=int, default=10) #if evaluation size == -1 takes all test_set


#Para teste no test set (test_and_error)

python main.py --scale_factor=2 --mode='test' --test_mode='pick_from_set' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 \
--num_channels=3 \
--batch_size=1 \
--image_size=64 \
--residual_loss_function='l2' \
--trained_discriminator=10000_cont_L1_resid_L1_DISC__loss_gen_0.212_loss_disc_0.001.pth.tar \
--trained_model=10000_cont_L1_resid_L1_GEN__loss_gen_0.212_loss_disc_0.001.pth.tar


#Para teste fora do test set (single_test) ---> batch_size = 1

python main.py --scale_factor=2 --mode='test' --test_mode='single' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=3 \
--batch_size=1 \
--image_size=64 \
--test_image_path='./test_images/185872.jpg_lr_50001.png' \
--trained_discriminator=10000_cont_L1_resid_L1_DISC__loss_gen_0.212_loss_disc_0.001.pth.tar \
--trained_model=10000_cont_L1_resid_L1_GEN__loss_gen_0.212_loss_disc_0.001.pth.tar


#Para muitos teste fora do test set (many_tests) -> passe o folder no test_image_path 

python main.py --scale_factor=2 --mode='test' --test_mode='many' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=3 \
--batch_size=1 \
--image_size=64 \
--test_image_path='./test_images/' \
--trained_discriminator=10000_cont_L1_resid_L1_DISC__loss_gen_0.212_loss_disc_0.001.pth.tar \
--trained_model=10000_cont_L1_resid_L1_GEN__loss_gen_0.212_loss_disc_0.001.pth.tar



#para evaluation (evaluate) - (does test_and_error many times) - Lembrar que aqui batch_size deve ser 1 (pra prever uma foto de cada vez) senão vai fazer predição pra várias fotos em um mesmo arquivo. 

###if evaluation size == -1 takes all test_set

python main.py --scale_factor=2 --mode='test' --test_mode='evaluate' --num_workers=0 --total_step=300000 --model_save_step=1000 --sample_step=100 --num_channels=3 --batch_size=1 --image_size=64 \
--trained_discriminator=10000_cont_L1_resid_L1_DISC__loss_gen_0.212_loss_disc_0.001.pth.tar \
--trained_model=10000_cont_L1_resid_L1_GEN__loss_gen_0.212_loss_disc_0.001.pth.tar \
--evaluation_step=10 \
--evaluation_size=3