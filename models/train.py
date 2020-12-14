import torch
import util


def train(model, optimizer, stats, run_args=None):
    device = model.device
    checkpoint_path = util.get_checkpoint_path(run_args)
    num_iterations_so_far = len(stats.losses)

    for iteration in range(num_iterations_so_far, run_args.num_iterations):
        # Loss
        loss = model(torch.rand(3, device=device), torch.rand((), device=device))

        # Backprop
        loss.backward()

        # Step
        optimizer.step()
        optimizer.zero_grad()

        stats.losses.append(loss.item())

        if iteration % run_args.log_interval == 0:
            util.logging.info(
                "it. {}/{} | loss = {:.2f}".format(iteration, run_args.num_iterations, loss)
            )

        if iteration % run_args.save_interval == 0:
            util.save_checkpoint(checkpoint_path, model, optimizer, stats, run_args=run_args)

        if iteration % run_args.checkpoint_interval == 0:
            util.save_checkpoint(
                util.get_checkpoint_path(run_args, checkpoint_iteration=iteration),
                model,
                optimizer,
                stats,
                run_args=run_args,
            )

    util.save_checkpoint(
        checkpoint_path, model, optimizer, stats, run_args=run_args,
    )
