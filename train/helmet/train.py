from rfdetr import RFDETRMedium

if __name__ == "__main__":
    model = RFDETRMedium()

    model.train(
        dataset_dir="train/helmet/datasets",
        epochs=30,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        num_workers=0,
        output_dir="train/helmet/output",
    )
